#!/usr/bin/env python3
"""
ROCm CI helper:
- Detect MI300 vs MI355
- Prefer wheel from this repo's 'te-rocm-wheels' release assets
- Fallback to pinned ROCm/maxtext release assets (arch-specific)

This script only downloads the wheel into the current working directory.
The caller should then install it (e.g. `uv pip install transformer_engine-*.whl`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.request


def _run(cmd: str) -> str:
  try:
    return subprocess.check_output(["bash", "-lc", cmd], text=True, stderr=subprocess.STDOUT)
  except (subprocess.CalledProcessError, FileNotFoundError, OSError):
    return ""


def detect_arch() -> str:
  """
  Return wheel arch selector: 'mi300' or 'mi355'.

  Notes:
  - MI300 family commonly reports gfx942/gfx941.
  - MI350/MI355 family commonly reports gfx950 and product strings like MI350X.
  """
  override = os.environ.get("TE_WHEEL_ARCH", "").strip().lower()
  if override in {"mi300", "mi355"}:
    return override

  rocm_smi = _run("command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showproductname || true") or _run(
      "[ -x /opt/rocm/bin/rocm-smi ] && /opt/rocm/bin/rocm-smi --showproductname || true"
  )
  rocminfo = _run("command -v rocminfo >/dev/null 2>&1 && rocminfo || true") or _run(
      "[ -x /opt/rocm/bin/rocminfo ] && /opt/rocm/bin/rocminfo || true"
  )

  blob = f"{rocm_smi}\n{rocminfo}".lower()
  gfxs = sorted(set(re.findall(r"gfx[0-9a-f]+", blob)))

  # Prefer explicit gfx IDs when available.
  if "gfx950" in gfxs:
    return "mi355"
  if "gfx942" in gfxs or "gfx941" in gfxs:
    return "mi300"

  # Fall back to product string checks.
  if "mi355" in blob or "mi350" in blob:
    return "mi355"
  if "mi300" in blob:
    return "mi300"

  # Safe default.
  return "mi355"


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


def try_download_from_te_rocm_wheels(repo: str, arch: str) -> bool:
  """Download from this repo's 'te-rocm-wheels' release tag if present."""
  api = f"https://api.github.com/repos/{repo}/releases/tags/te-rocm-wheels"
  req = urllib.request.Request(api, headers=_headers())
  with urllib.request.urlopen(req) as r:
    rel = json.loads(r.read().decode("utf-8"))

  assets = rel.get("assets", [])
  name_re = re.compile(rf"^transformer_engine-.*-1\.{arch}-cp312-cp312-linux_x86_64\.whl$")
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
      help="Print resolved wheel arch (mi300/mi355) and exit.",
  )
  args = parser.parse_args(argv)

  if args.print_arch:
    print(detect_arch(), flush=True)
    return 0

  repo = os.environ.get("GITHUB_REPOSITORY", "")
  if not repo:
    print("[te wheel] GITHUB_REPOSITORY not set; skipping.", flush=True)
    return 0

  arch = detect_arch()
  print(f"[te wheel] arch={arch}", flush=True)

  # 1) Prefer: this repo's te-rocm-wheels assets.
  try:
    if try_download_from_te_rocm_wheels(repo, arch):
      return 0
    print(f"[te wheel] no te-rocm-wheels asset for arch={arch}", flush=True)
  except urllib.error.HTTPError as e:
    print(f"[te wheel] te-rocm-wheels not available ({e.code})", flush=True)
  except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as e:
    print(f"[te wheel] te-rocm-wheels lookup failed ({e})", flush=True)

  # 2) Fallback: pinned ROCm/maxtext release assets.
  arch_tag = f"1.{arch}"
  pinned_name = f"transformer_engine-2.8.0.dev0+2776c337-{arch_tag}-cp312-cp312-linux_x86_64.whl"
  pinned = "https://github.com/ROCm/maxtext/releases/download/rocm-maxtext-v0.1.1/" f"{pinned_name}"
  download(pinned, pinned_name)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
