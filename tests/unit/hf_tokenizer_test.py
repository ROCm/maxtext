"""The paged path's tokenizer must need neither JetStream nor torch.

Both halves matter and for different reasons. JetStream was archived on
2026-02-01, so `MaxEngine.build_tokenizer` -- which requires it even for
HuggingFace tokenizers -- is a dependency that has stopped moving. And
`transformers` imports torch when it finds it, which gives the process a second
HIP runtime and aborts RCCL clique setup above one device, so the obvious
replacement is worse than the problem.

The two assertions that carry the point are the ones checking `sys.modules`.
Everything else here is ordinary behaviour that would be caught by any use.

Copyright 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from maxtext.inference import hf_tokenizer


def _tiny_tokenizer_json() -> dict:
  """A minimal word-level `tokenizer.json`, so this test needs no checkpoint.

  Word-level rather than BPE because the point is the wrapper, not the encoding:
  a vocabulary small enough to read makes an assertion about ids legible.
  """
  vocab = {"hello": 0, "world": 1, "<eos>": 2, "<pad>": 3}
  return {
      "version": "1.0",
      "truncation": None,
      "padding": None,
      "added_tokens": [
          {
              "id": 2,
              "content": "<eos>",
              "single_word": False,
              "lstrip": False,
              "rstrip": False,
              "normalized": False,
              "special": True,
          },
          {
              "id": 3,
              "content": "<pad>",
              "single_word": False,
              "lstrip": False,
              "rstrip": False,
              "normalized": False,
              "special": True,
          },
      ],
      "normalizer": None,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": None,
      "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": False},
      "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<pad>"},
  }


class HfTokenizerTest(unittest.TestCase):
  """Behaviour, and the absence of two dependencies."""

  def setUp(self):
    super().setUp()
    self.dir = tempfile.mkdtemp()
    with open(os.path.join(self.dir, "tokenizer.json"), "wt", encoding="utf-8") as handle:
      json.dump(_tiny_tokenizer_json(), handle)
    with open(os.path.join(self.dir, "tokenizer_config.json"), "wt", encoding="utf-8") as handle:
      json.dump({"eos_token": "<eos>", "pad_token": "<pad>"}, handle)

  def test_it_round_trips_and_resolves_special_ids(self):
    tokenizer = hf_tokenizer.build_tokenizer(self.dir)
    self.assertEqual(tokenizer.encode("hello world"), [0, 1])
    self.assertEqual(tokenizer.eos_id, 2, "eos resolved from tokenizer_config, not guessed")
    self.assertEqual(tokenizer.pad_id, 3)
    self.assertIn("hello", tokenizer.decode([0, 1]))

  def test_a_dict_shaped_special_token_entry_resolves(self):
    """Checkpoints export these as a bare string or as a dict; both appear."""
    with open(os.path.join(self.dir, "tokenizer_config.json"), "wt", encoding="utf-8") as handle:
      json.dump({"eos_token": {"content": "<eos>", "lstrip": False}}, handle)
    self.assertEqual(hf_tokenizer.build_tokenizer(self.dir).eos_id, 2)

  def test_an_explicit_eos_overrides_the_checkpoint(self):
    """A base model used with instruction formatting stops on a different token."""
    self.assertEqual(hf_tokenizer.build_tokenizer(self.dir, eos_id=1).eos_id, 1)

  def test_decode_accepts_numpy_and_nested_shapes(self):
    """Generated tokens arrive as arrays, sometimes with a leading batch axis."""
    tokenizer = hf_tokenizer.build_tokenizer(self.dir)
    self.assertEqual(tokenizer.decode(np.asarray([0, 1])), tokenizer.decode([0, 1]))
    self.assertEqual(tokenizer.decode(np.asarray([[0, 1]])), tokenizer.decode([0, 1]))

  def test_encode_adds_no_special_tokens_by_default(self):
    """A silently prepended BOS shifts every absolute position by one.

    The paged path positions tokens absolutely and RoPE is not translation
    invariant, so this default is load-bearing rather than a preference.
    """
    tokenizer = hf_tokenizer.build_tokenizer(self.dir)
    self.assertEqual(len(tokenizer.encode("hello world")), 2)

  def test_a_missing_tokenizer_json_fails_loudly(self):
    """Falling back to transformers here would trade a missing file for an RCCL abort."""
    with tempfile.TemporaryDirectory() as empty:
      with self.assertRaises(FileNotFoundError):
        hf_tokenizer.build_tokenizer(empty)

  def test_an_unresolvable_eos_fails_rather_than_defaulting(self):
    """Without an EOS every request runs to its length cap, which reads as a quality bug."""
    with open(os.path.join(self.dir, "tokenizer_config.json"), "wt", encoding="utf-8") as handle:
      json.dump({}, handle)
    with self.assertRaises(ValueError):
      hf_tokenizer.build_tokenizer(self.dir)

  def test_it_imports_neither_jetstream_nor_transformers_nor_torch(self):
    """The whole point, checked in a fresh interpreter.

    In-process this would be unreliable: another test may already have imported
    one of these, and `sys.modules` is global. A subprocess is the only honest way
    to assert what *this* module pulls in.

    **`transformers` is on the list, and leaving it off made this test useless.**
    A first version forbade only `torch` and `jetstream`, and it passed with an
    `import transformers` deliberately added to the module under test -- because
    transformers imports torch *lazily*, so nothing named `torch` is in
    `sys.modules` at import time. The hazard is reaching for transformers at all:
    it probes for torch with `find_spec` and imports it when found, at which point
    a second HIP runtime aborts RCCL clique setup. Forbidding the proximate cause
    rather than the symptom is what makes the check bite.
    """
    script = (
        "import sys, json, os, tempfile\n"
        "sys.path.insert(0, os.environ['MAXTEXT_SRC'])\n"
        f"d = {self.dir!r}\n"
        "from maxtext.inference import hf_tokenizer\n"
        "t = hf_tokenizer.build_tokenizer(d)\n"
        "assert t.encode('hello world') == [0, 1]\n"
        "assert t.decode([0, 1])\n"
        "forbidden = ('torch', 'jetstream', 'transformers')\n"
        # Top-level names only. Reporting every submodule turns a one-line
        # diagnosis into fourteen kilobytes of `transformers.models.*`.
        "bad = {m.split('.')[0] for m in sys.modules if m.split('.')[0] in forbidden}\n"
        "print('LEAKED:' + ','.join(sorted(bad)))\n"
    )
    env = dict(os.environ)
    env["MAXTEXT_SRC"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env
    )
    leaked = [line for line in out.stdout.splitlines() if line.startswith("LEAKED:")]
    self.assertEqual(leaked, ["LEAKED:"], f"the tokenizer pulled in a forbidden package: {out.stdout}")


if __name__ == "__main__":
  unittest.main()
