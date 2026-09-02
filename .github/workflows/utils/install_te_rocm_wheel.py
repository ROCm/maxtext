#!/usr/bin/env python3
"""
ROCm CI helper:
- Resolve the MI355 Transformer Engine wheel from this repo's
  'te-rocm-wheels' release assets, falling back to the pinned ROCm/maxtext
  release asset if needed.

CI runners are MI355 only, so no architecture detection is performed.

This script only downloads the wheel into the current working directory.
The caller should then install it (e.g. `uv pip install transformer_engine-*.whl`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request


# CI runners are always MI355 (gfx950); no detection or fallback needed.
WHEEL_ARCH = "mi355"


def _headers() -> dict[str, str]:
  token = os.environ.get("GITHUB_TOKEN", "")
  headers: dict[str, str] = {
      "Accept": "application/vnd.github+json",
      "User-Agent": "maxtext-ci",
  }
  if token:
    headers["Authorization"] = f"Bearer {token}"
  return headers


def download(url: str, out_name: str) -> None:
  req = urllib.request.Request(url, headers=_headers())
  with urllib.request.urlopen(req) as r, open(out_name, "wb") as f:
    f.write(r.read())
  print(f"[te wheel] downloaded {out_name}", flush=True)


def try_download_from_te_rocm_wheels(repo: str) -> bool:
  """Download from this repo's 'te-rocm-wheels' release tag if present."""
  api = f"https://api.github.com/repos/{repo}/releases/tags/te-rocm-wheels"
  req = urllib.request.Request(api, headers=_headers())
  with urllib.request.urlopen(req) as r:
    rel = json.loads(r.read().decode("utf-8"))

  assets = rel.get("assets", [])
  name_re = re.compile(rf"^transformer_engine-.*-1\.{WHEEL_ARCH}-cp312-cp312-linux_x86_64\.whl$")
  matches = [a for a in assets if name_re.match(a.get("name", ""))]
  if not matches:
    return False

  # Rolling tag keeps many wheels; select newest matching asset.
  hit = max(matches, key=lambda a: a.get("created_at", ""))
  print(
      "[te wheel] selected latest te-rocm-wheels asset: "
      f"{hit.get('name', '<unknown>')} (created_at={hit.get('created_at', 'unknown')})",
      flush=True,
  )

  download(hit["browser_download_url"], hit["name"])
  return True


def main(argv: list[str] | None = None) -> int:
  """Entry point."""
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument(
      "--print-arch",
      action="store_true",
      help="Print the wheel arch (always 'mi355') and exit.",
  )
  args = parser.parse_args(argv)

  if args.print_arch:
    print(WHEEL_ARCH, flush=True)
    return 0

  repo = os.environ.get("GITHUB_REPOSITORY", "")
  if not repo:
    print("[te wheel] GITHUB_REPOSITORY not set; skipping.", flush=True)
    return 0

  print(f"[te wheel] arch={WHEEL_ARCH}", flush=True)

  # 1) Prefer: this repo's te-rocm-wheels assets.
  try:
    if try_download_from_te_rocm_wheels(repo):
      return 0
    print(f"[te wheel] no te-rocm-wheels asset for arch={WHEEL_ARCH}", flush=True)
  except urllib.error.HTTPError as e:
    print(f"[te wheel] te-rocm-wheels not available ({e.code})", flush=True)
  except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as e:
    print(f"[te wheel] te-rocm-wheels lookup failed ({e})", flush=True)

  # 2) Fallback: pinned ROCm/maxtext release asset.
  pinned_name = f"transformer_engine-2.8.0.dev0+2776c337-1.{WHEEL_ARCH}-cp312-cp312-linux_x86_64.whl"
  pinned = f"https://github.com/ROCm/maxtext/releases/download/rocm-maxtext-v0.1.1/{pinned_name}"
  download(pinned, pinned_name)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
