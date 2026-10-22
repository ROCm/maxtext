# Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`AttentionOp._flydsl_flash_packed`: segment ids -> document boundaries -> variable-length flash attention.

Checks the MaxText wiring (not the kernel, which has its own tests): rows of packed documents followed by padding
(segment id 0) must give each real token exactly the attention of its own document, padded tokens must not leak into
real ones, and gradients must match float32 per-document attention. Runs on one GPU with a stand-in for the attention
object, since the method only uses `config.max_segments_per_seq` and `mesh`.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

pytest.importorskip("jax_flydsl", reason="requires jax-flydsl on PYTHONPATH")

from maxtext.layers.attention_op import AttentionOp  # pylint: disable=wrong-import-position

HEADS, DQK, DV, SEQ = 4, 192, 128, 1024
MAX_SEGMENTS = 8


def _no_batch_axes():
  return ()  # one device: the batch is not sharded


def _fake_op():
  """The stand-in `self`: the method only reads config.max_segments_per_seq, mesh and _flydsl_batch_axes()."""
  mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
  return SimpleNamespace(
      config=SimpleNamespace(max_segments_per_seq=MAX_SEGMENTS), mesh=mesh, _flydsl_batch_axes=_no_batch_axes
  )


def _segment_ids(doc_lengths_per_row):
  """[rows, SEQ] ids: 1..n for each row's documents, then 0 for padding."""
  rows = []
  for lens in doc_lengths_per_row:
    ids = np.concatenate([np.full(n, i + 1, np.int32) for i, n in enumerate(lens)] + [np.zeros(SEQ - sum(lens), np.int32)])
    rows.append(ids)
  return np.stack(rows)


def _ref_doc(q, k, v, scale):
  s = jnp.einsum("shd,thd->hst", q, k) * scale
  t = q.shape[0]
  p = jax.nn.softmax(jnp.where(jnp.tril(jnp.ones((t, t), bool)), s, -jnp.inf), axis=-1)
  return jnp.einsum("hst,thd->shd", p, v)


def _run(doc_lengths_per_row, seed=0):
  """Output and gradients of the wiring and of per-document float32 attention, for rows of the given documents."""
  rows = len(doc_lengths_per_row)
  seg = jnp.asarray(_segment_ids(doc_lengths_per_row))
  rng = np.random.default_rng(seed)

  def mk(d):
    return jnp.asarray(rng.standard_normal((rows, SEQ, HEADS, d)), jnp.bfloat16)

  q, k, v = mk(DQK), mk(DQK), mk(DV)
  real = (seg > 0)[..., None, None]
  cot = jnp.where(real, jnp.asarray(rng.standard_normal((rows, SEQ, HEADS, DV)), jnp.bfloat16), 0)  # the loss masks padding
  op = _fake_op()

  def fwd(a, b, c):
    return AttentionOp._flydsl_flash_packed(op, a, b, c, seg)  # pylint: disable=protected-access

  out, vjp = jax.vjp(fwd, q, k, v)
  dq, dk, dv = vjp(cot)

  def ref(a, b, c):
    outs = []
    for r, lens in enumerate(doc_lengths_per_row):
      o = jnp.zeros((SEQ, HEADS, DV), jnp.float32)
      start = 0
      for n in lens:
        o = o.at[start : start + n].set(
            _ref_doc(a[r, start : start + n], b[r, start : start + n], c[r, start : start + n], 1.0)
        )
        start += n
      outs.append(o)
    return jnp.stack(outs)

  ref_out, ref_vjp = jax.vjp(ref, *(x.astype(jnp.float32) for x in (q, k, v)))
  ref_dq, ref_dk, ref_dv = ref_vjp(cot.astype(jnp.float32))
  return seg, out, (dq, dk, dv), ref_out, (ref_dq, ref_dk, ref_dv)


def _rel(got, want, mask):
  """Relative Frobenius error over the masked (real-token) elements."""
  g = np.asarray(got, np.float32)[mask]
  w = np.asarray(want, np.float32)[mask]
  return np.linalg.norm(g - w) / (np.linalg.norm(w) + 1e-30)


@pytest.mark.parametrize(
    "docs",
    [
        [[SEQ]],  # one full row
        [[300, 500, 100], [1000]],  # padding after the documents; one full row of one document
        [[1, 2, 3, 4, 5, 6, 7, 8]],  # the limit: 8 documents plus padding is 9 runs
        [[100] * 8, [64, 64, 64, 64]],  # exactly max_segments documents, then fewer
    ],
    ids=["full_row", "padding_and_full", "eight_tiny_docs", "at_limit"],
)
def test_matches_per_document_reference(docs):
  seg, out, grads, ref_out, ref_grads = _run(docs)
  real = np.asarray(seg > 0)[..., None, None] * np.ones((1, 1, HEADS, 1), bool)
  e_out = _rel(out, ref_out, np.broadcast_to(real, out.shape))
  errs = [
      _rel(g, r, np.broadcast_to(real[..., :1] * np.ones((1, 1, 1, g.shape[-1]), bool), g.shape))
      for g, r in zip(grads, ref_grads)
  ]
  print(
      f"ACC|packed wiring docs={[len(d) for d in docs]}|out={e_out:.4f} dq={errs[0]:.4f} dk={errs[1]:.4f} dv={errs[2]:.4f}"
  )
  assert not np.isnan(np.asarray(out, np.float32)).any()
  assert e_out < 0.02 and max(errs) < 0.03, (e_out, errs)
  # Padding tokens carry no cotangent, so they must receive none: a leak across the padding boundary would show here.
  pad = np.asarray(seg == 0)
  if pad.any():
    for g in grads:
      assert np.abs(np.asarray(g, np.float32)[pad]).max() == 0.0, "gradient reached a padding token"


def test_more_documents_than_max_segments_is_a_known_limit():
  """Beyond max_segments_per_seq documents in a row, runs past the last slot merge: attention crosses documents.

  This documents the limit (the code comment calls it a shortcut) rather than asserting correctness; the test fails only if
  the merge stops being visible, which would mean the limit changed and the comment is stale.
  """
  docs = [[50] * 12]  # 12 documents + padding = 13 runs, more than the 9 slots (8 documents + padding)
  seg, out, _, ref_out, _ = _run(docs)
  mask = np.broadcast_to(np.asarray(seg > 0)[..., None, None], out.shape)
  err = _rel(out, ref_out, mask)
  print(f"ACC|packed wiring beyond limit docs=12 max_segments={MAX_SEGMENTS}|out={err:.4f}")
  assert err > 0.02, "documents past max_segments_per_seq no longer merge; update the shortcut comment"
