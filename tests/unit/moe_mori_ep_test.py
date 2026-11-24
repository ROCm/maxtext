# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parity test for the MORI expert-parallel routed-MoE path (moe_use_mori_ep).

This exercises ``RoutedMoE._mori_ep_matmul`` end-to-end against a plain-JAX
reference MoE computed from the same gated routing and expert weights, and
asserts the two agree within bf16 tolerance.

The MORI FFI needs its symmetric-memory runtime and a static per-process rank,
so (like ``mori``'s own JAX tests) this spins up one worker process per GPU with
``jax.distributed`` + ``mori.jax.shmem_init_attr``. It is therefore skipped by
default and only runs when ``MORI_MAXTEXT_TEST=1`` and the ``mori`` package is
importable. Expected environment (single node): one GPU per process, a pure
expert-parallel mesh spanning every process.

Run (inside the ROCm container, adapting Urun_pytest.sh for LD_LIBRARY_PATH /
MORI_KERNEL_DIR)::

    MORI_MAXTEXT_TEST=1 MORI_NUM_PROCS=8 MORI_TOTAL_GPUS=8 \
        python3 -m pytest tests/unit/moe_mori_ep_test.py -s -v
"""
import os
import multiprocessing
import socket
import traceback

import pytest

_RUN = os.environ.get("MORI_MAXTEXT_TEST", "0") == "1"
_NUM_PROCS = int(os.environ.get("MORI_NUM_PROCS", "8"))
_TOTAL_GPUS = int(os.environ.get("MORI_TOTAL_GPUS", "8"))
_WORKER_TIMEOUT = int(os.environ.get("MORI_TEST_TIMEOUT", "600"))

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

# gpu_only is required: tests with no hardware marker are auto-marked cpu_only
# and skipped on accelerator testbeds (tests/conftest.py).
pytestmark = [
    pytest.mark.gpu_only,
    pytest.mark.skipif(
        not _RUN, reason="Set MORI_MAXTEXT_TEST=1 (needs ROCm + the mori package) to run."
    ),
]


# ---------------------------------------------------------------------------
# Multi-process harness (mirrors mori/tests/python/ops/test_dispatch_combine_jax.py)
# ---------------------------------------------------------------------------


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, coordinator_port, total_gpus, result_queue):
    """One GPU per process: init distributed + MORI shmem, run the parity check."""
    gpus_per_proc = total_gpus // world_size
    local_devices = [rank * gpus_per_proc + i for i in range(gpus_per_proc)]

    try:
        import jax

        # Every process sees every GPU and JAX takes its own. RCCL on ROCm cannot
        # build its intra-node graph ("TopoPreset: Local rank N not found") when
        # HIP_VISIBLE_DEVICES leaves each process one GPU, and the first XLA
        # collective (the parity allgather) then fails.
        jax.distributed.initialize(
            coordinator_address=f"localhost:{coordinator_port}",
            num_processes=world_size,
            process_id=rank,
            local_device_ids=local_devices,
        )

        import mori
        from mori.jit.hip_driver import _check, _get_hip_lib

        # mori runs on the calling thread's HIP device; with every GPU visible,
        # that has to be this process's own.
        _check(_get_hip_lib().hipSetDevice(local_devices[0]), "hipSetDevice")
        mori.jax.shmem_init_attr(rank, world_size)

        _run_parity(rank, world_size)
        result_queue.put((rank, None))
    except Exception:  # pylint: disable=broad-exception-caught
        result_queue.put((rank, traceback.format_exc()))
    finally:
        try:
            import mori

            mori.jax.shmem_finalize()
        except Exception:  # pylint: disable=broad-exception-caught
            pass


def _run_parity(rank, world_size):
    """Build a small RoutedMoE with moe_use_mori_ep=True on a pure-EP mesh and
    compare its output to a local plain-JAX reference MoE."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning

    from jax.experimental.multihost_utils import process_allgather

    from maxtext.configs import pyconfig
    from maxtext.layers import moe
    from maxtext.layers.initializers import nd_dense_init
    from maxtext.utils import maxtext_utils
    from tests.utils.test_helpers import get_test_config_path

    def _to_host(x):
        """Full numpy value of x.

        With one GPU per process the MoE output and expert weights are global
        arrays: this process only owns its shard, so np.asarray raises.
        process_allgather(tiled=True) reshardes to replicated and returns the
        whole array. Fully local arrays are copied as-is.
        """
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            return np.asarray(process_allgather(x, tiled=True))
        return np.asarray(x)

    seq = 8
    emb = 256
    mlp = 512
    num_experts = 2 * world_size
    top_k = 4

    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name="moe_mori_ep_parity",
        enable_checkpointing=False,
        dtype="bfloat16",
        megablox=False,
        sparse_matmul=True,
        moe_use_mori_ep=True,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        base_emb_dim=emb,
        base_mlp_dim=mlp,
        base_moe_mlp_dim=mlp,
        max_target_length=seq,
        max_prefill_predict_length=seq,
        per_device_batch_size=1,
        ici_expert_parallelism=world_size,
    )

    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = moe.RoutedMoE(
        config=cfg,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        mesh=mesh,
        kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", "mlp"),
        dtype=cfg.dtype,
        rngs=nnx.Rngs(params=0),
    )

    batch = world_size  # per_device_batch_size=1 over world_size devices
    assert batch % model.get_expert_parallelism_size() == 0

    # Deterministic global input, identical on every process.
    full_x = np.asarray(
        jax.random.normal(jax.random.PRNGKey(123), (batch, seq, emb), dtype=jnp.float32)
    ).astype(jnp.bfloat16)
    x_sharding = NamedSharding(mesh, P())  # replicated; shard_map reshards internally
    inputs = jax.make_array_from_callback(full_x.shape, x_sharding, lambda idx: full_x[idx])

    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
        output, _, _ = model(inputs)
        # Same global inputs the forward used. A single-device copy of full_x
        # cannot be dotted with the expert-sharded gate kernel.
        gate_logits, pre_bias_logits = model.gate(inputs)
    mori_out = _to_host(output).astype(np.float32)

    # -- Reference: full MoE computed locally from the same routing + weights --
    x2 = jnp.asarray(full_x).reshape(batch * seq, emb)
    gate_logits = jnp.asarray(_to_host(gate_logits))
    if pre_bias_logits is not None:
        pre_bias_logits = jnp.asarray(_to_host(pre_bias_logits))
    weights, indices = model.get_topk(gate_logits, pre_bias_logits, rngs=None)
    weights = _to_host(weights).reshape(batch * seq, top_k).astype(np.float32)
    indices = _to_host(indices).reshape(batch * seq, top_k)

    wi0 = jnp.asarray(_to_host(model.wi_0[...]))  # [E, emb, mlp]
    wi1 = jnp.asarray(_to_host(model.wi_1[...]))
    wo = jnp.asarray(_to_host(model.wo[...]))  # [E, mlp, emb]

    ref = jnp.zeros((batch * seq, emb), jnp.float32)
    for e in range(num_experts):
        layer_w0 = x2.astype(cfg.dtype) @ wi0[e]
        layer_w1 = x2.astype(cfg.dtype) @ wi1[e]
        inter = model.apply_ffn_activation(layer_w0, layer_w1)
        out_e = (inter @ wo[e]).astype(jnp.float32)
        sel = indices == e
        w_e = jnp.sum(weights * sel, axis=-1)
        ref = ref + w_e[:, None] * out_e
    ref = np.asarray(ref).reshape(batch, seq, emb)

    diff = np.abs(mori_out - ref)
    tol = 3e-2 + 3e-2 * np.abs(ref)
    n_bad = int(np.sum(diff > tol))
    print(
        f"rank {rank}: max_abs_diff={float(diff.max()):.4e} mismatched={n_bad}/{ref.size}",
        flush=True,
    )
    assert n_bad == 0, (
        f"rank {rank}: MORI EP MoE output differs from reference "
        f"(max_abs_diff={float(diff.max()):.4e}, {n_bad} mismatched)"
    )


def test_moe_mori_ep_parity():
    """Spawn one process per GPU and assert MORI-EP parity on all of them."""
    try:
        import mori  # noqa: F401
    except ImportError:
        pytest.skip("mori package not importable")

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    port = _get_free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, _NUM_PROCS, port, _TOTAL_GPUS, result_queue))
        for rank in range(_NUM_PROCS)
    ]
    for p in procs:
        p.start()

    failures = []
    for _ in range(_NUM_PROCS):
        rank, tb = result_queue.get(timeout=_WORKER_TIMEOUT)
        if tb is not None:
            failures.append((rank, tb))
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    for r, tb in failures:
        print(f"\n=== Rank {r} FAILED ===\n{tb}")
    if failures:
        pytest.fail(f"{len(failures)} rank(s) failed: " + ", ".join(str(r) for r, _ in failures))
