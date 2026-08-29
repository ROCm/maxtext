"""Cache identity: everything that can change the K/V produced for the same tokens.

A prefix cache answers "have I already computed the K/V for these token ids?".
That question is only well posed relative to everything *else* that determines the
answer, and the failure mode when the set is incomplete is the worst kind: a
correct-looking cache hit that returns another configuration's K/V. Nothing
crashes, nothing is logged, and the output is merely wrong.

So the namespace is folded into the block hash chain rather than compared
alongside it. Two requests in different namespaces do not traverse the same
subtree at all, because their very first block hash differs. A mismatch cannot
produce a hit, as opposed to producing one that a later check is relied upon to
catch.

**The digest iterates the dataclass fields rather than listing them.** That is the
load-bearing detail. A hand-written digest is a second place to remember every
field, and the one time someone adds a field and forgets is the one time two
incompatible configurations collide. Adding a field here puts it in the hash
automatically.

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

import dataclasses
import hashlib

CACHE_NAMESPACE_VERSION = 1


@dataclasses.dataclass(frozen=True)
class CacheNamespace:
  """Identity of the configuration that produced a cached K/V page.

  Every field is a string so the digest is defined without per-type handling, and
  so a caller can put whatever summary is meaningful in it -- a checkpoint hash, a
  serialised RoPE config, a digest of image inputs. What matters is that two
  configurations which would produce different K/V produce different strings.

  Attributes:
    model_fingerprint: identity of the weights themselves, ideally a checkpoint
      digest rather than a name. Two finetunes of one base model share a name and
      must not share a cache.
    model_revision: the version of those weights, for the case where the
      fingerprint is a mutable reference.
    tokenizer: tokenizer identity. The same text maps to different ids under a
      different tokenizer, and the same ids to different text.
    adapter: LoRA or other adapter identity. An adapter changes the projections
      and therefore the K/V for identical tokens.
    tenant: cache domain. Not a correctness field but an isolation one -- two
      tenants may be unwilling to share pages even when sharing is sound.
    rope: RoPE configuration. Scaling factors and base frequency change the K/V
      for a token at a given position, and llama3-style scaling makes this a real
      variant rather than a theoretical one.
    kv_dtype: storage dtype of the pool.
    kv_quantization: quantisation scheme and scales, if any.
    layout: page size and physical form. A page cached under one layout is not
      readable under another.
    sharding: tensor-parallel width and axis mapping. Under replication the same
      logical head lives on several devices, and a page cached at one TP width is
      not valid at another.
    prompt_embeddings: digest of any soft-prompt or prefix-tuning embeddings,
      which change the K/V without changing a single token id.
    multimodal: digest of image, audio or video inputs interleaved with the text,
      for the same reason.
  """

  model_fingerprint: str = ""
  model_revision: str = ""
  tokenizer: str = ""
  adapter: str = ""
  tenant: str = ""
  rope: str = ""
  kv_dtype: str = ""
  kv_quantization: str = ""
  layout: str = ""
  sharding: str = ""
  prompt_embeddings: str = ""
  multimodal: str = ""
  version: int = CACHE_NAMESPACE_VERSION

  def digest(self) -> bytes:
    """A stable 32-byte digest over *every* field of this dataclass.

    Field names are hashed alongside their values, so renaming a field or
    reordering the declaration changes the digest. That is deliberate: either
    change means the stored meaning has moved, and silently keeping old entries
    valid across it would be a correctness bug rather than a convenience.
    """
    hasher = hashlib.blake2b(digest_size=32)
    for field in dataclasses.fields(self):
      hasher.update(field.name.encode("utf-8"))
      hasher.update(b"\x00")
      hasher.update(str(getattr(self, field.name)).encode("utf-8"))
      hasher.update(b"\x01")
    return hasher.digest()

  def describe(self) -> str:
    """The non-empty fields, for a log line that explains a cache miss."""
    parts = [
        f"{f.name}={getattr(self, f.name)}"
        for f in dataclasses.fields(self)
        if f.name != "version" and getattr(self, f.name)
    ]
    return ", ".join(parts) if parts else "<all defaults>"
