"""A tokenizer for the paged inference path that needs neither JetStream nor torch.

`MaxEngine.build_tokenizer` is the ordinary route and it is unavailable to a
paged-only deployment for two independent reasons.

**It requires JetStream, including for HuggingFace tokenizers.** It raises outright
under `DECOUPLE_GCLOUD=TRUE`, and even its `huggingface` branch returns
`jetstream.engine.token_utils.HuggingFaceTokenizer` -- so the tokenizer type does
not change the dependency. JetStream was archived on 2026-02-01, with its
functionality migrated into `vllm-project/tpu-inference`, so this is a dependency
that has stopped moving rather than one to wait on. The rest of the paged path
already runs under the `DECOUPLE_GCLOUD` stubs; tokenisation was the last tie.

**And the obvious replacement carries a worse problem.** `transformers.AutoTokenizer`
works, and every scratch script in this project uses it, but `transformers` probes
for torch with `importlib.util.find_spec` and imports it when found. Torch brings
its own bundled ROCm, giving the process a second HIP runtime, and
`rocprofiler-register` then aborts during RCCL clique setup -- which is fatal above
one device. The scratch scripts get away with it by installing a `find_spec` shim
before importing anything, and library code should not have to.

So this uses `tokenizers` directly. It is the same Rust implementation
`transformers` wraps, reads the same `tokenizer.json`, and pulls in no torch:
verified by `"torch" not in sys.modules` after a full encode and decode.

**The surface is defined by MaxText's own callers, not by guesswork.** Across
`maxtext/inference/` those are `eos_id`, `encode`, `decode`, and the underlying
HuggingFace tokenizer for `apply_chat_template` and `batch_decode` -- which
JetStream's wrapper also exposes as `.tokenizer`, so reaching through works the
same either way.

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

from __future__ import annotations

import json
import os
from typing import Any, Sequence


class HuggingFaceTokenizer:
  """MaxText's tokenizer surface over a `tokenizers.Tokenizer`.

  Duck-typed against JetStream's wrapper rather than subclassing it, for the same
  reason the paged attention path duck-types vLLM metadata: importing the thing
  you are trying not to depend on defeats the exercise.
  """

  def __init__(self, tokenizer: Any, *, eos_id: int, pad_id: int | None = None):
    self._tokenizer = tokenizer
    self._eos_id = int(eos_id)
    self._pad_id = int(pad_id) if pad_id is not None else int(eos_id)

  @property
  def tokenizer(self) -> Any:
    """The underlying HuggingFace tokenizer.

    JetStream's wrapper exposes the same attribute, so callers reaching for
    `apply_chat_template` or `batch_decode` work unchanged.
    """
    return self._tokenizer

  @property
  def eos_id(self) -> int:
    return self._eos_id

  @property
  def pad_id(self) -> int:
    return self._pad_id

  def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
    """Text to ids.

    `add_special_tokens` defaults to False because the inference path positions
    tokens absolutely and a silently prepended BOS shifts every position by one.
    Callers that want the chat template should reach it through `.tokenizer`,
    where the choice is explicit.
    """
    return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

  def decode(self, ids: Sequence[int]) -> str:
    """Ids to text, tolerating numpy arrays and nested single-row shapes."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    flat = np.asarray(ids).reshape(-1).tolist()
    return self._tokenizer.decode([int(i) for i in flat])


def _resolve_special_id(tokenizer: Any, config: dict, key: str) -> int | None:
  """Look up a special token's id from `tokenizer_config.json`.

  The entry is either a plain string or a dict with a `content` field, depending
  on how the checkpoint was exported; both appear in the wild.
  """
  entry = config.get(key)
  if entry is None:
    return None
  token = entry.get("content") if isinstance(entry, dict) else entry
  if not isinstance(token, str):
    return None
  return tokenizer.token_to_id(token)


def build_tokenizer(tokenizer_path: str, *, eos_id: int | None = None) -> HuggingFaceTokenizer:
  """Load a tokenizer from a local HuggingFace checkpoint directory or file.

  Args:
    tokenizer_path: a directory holding `tokenizer.json`, or the file itself.
    eos_id: overrides what the checkpoint declares. Supply it when a deployment
      stops on a different token than the checkpoint's default, which is common
      for base models used with instruction formatting.

  Returns:
    A `HuggingFaceTokenizer`.

  Raises:
    FileNotFoundError: when no `tokenizer.json` is present. That file is what
      makes this torch-free, so falling back to `transformers` would trade a
      missing file for an RCCL abort above one device -- a worse failure, and a
      much less obvious one.
    ValueError: when no EOS id can be determined, since generation would then run
      to the length cap on every request and look like a quality problem.
  """
  path = tokenizer_path
  if os.path.isdir(path):
    candidate = os.path.join(path, "tokenizer.json")
  else:
    candidate = path
  if not os.path.isfile(candidate):
    raise FileNotFoundError(
        f"no tokenizer.json at {candidate!r}. The paged path loads tokenizers through the `tokenizers` "
        f"package to stay free of both JetStream and torch, and that needs the fast-tokenizer file. "
        f"Convert the tokenizer, or pass a ready-made tokenizer object instead."
    )

  # pylint: disable=import-outside-toplevel
  from tokenizers import Tokenizer

  tokenizer = Tokenizer.from_file(candidate)

  config: dict = {}
  config_path = os.path.join(os.path.dirname(candidate), "tokenizer_config.json")
  if os.path.isfile(config_path):
    with open(config_path, "rt", encoding="utf-8") as handle:
      config = json.load(handle)

  resolved = eos_id if eos_id is not None else _resolve_special_id(tokenizer, config, "eos_token")
  if resolved is None:
    raise ValueError(
        f"could not determine an EOS id from {config_path!r}. Pass eos_id explicitly; without it every "
        f"request generates to its length cap, which reads as a model quality problem rather than a "
        f"configuration one."
    )
  pad = _resolve_special_id(tokenizer, config, "pad_token")
  return HuggingFaceTokenizer(tokenizer, eos_id=int(resolved), pad_id=pad)
