#!/usr/bin/env python3
"""
Publish and prune ROCm TransformerEngine wheel assets on GitHub Releases.

Intended for use in GitHub Actions. Requires:
- GITHUB_TOKEN
- GITHUB_REPOSITORY (owner/repo)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"


def _env_token() -> str:
  token = os.environ.get("GITHUB_TOKEN")
  if not token:
    raise SystemExit("GITHUB_TOKEN is not set.")
  return token


def _env_repo() -> tuple[str, str]:
  repo = os.environ.get("GITHUB_REPOSITORY")
  if not repo or "/" not in repo:
    raise SystemExit("GITHUB_REPOSITORY is not set (expected 'owner/repo').")
  owner, name = repo.split("/", 1)
  return owner, name


def _headers(token: str) -> dict[str, str]:
  return {
      "Authorization": f"Bearer {token}",
      "Accept": "application/vnd.github+json",
      "User-Agent": "maxtext-ci",
  }


def _request_json(method: str, url: str, token: str, body: dict | None = None):
  data = None
  if body is not None:
    data = json.dumps(body).encode("utf-8")
  req = urllib.request.Request(url, data=data, method=method, headers=_headers(token))
  with urllib.request.urlopen(req) as r:
    raw = r.read()
    return json.loads(raw.decode("utf-8")) if raw else None


def _request_raw(method: str, url: str, token: str, data: bytes, content_type: str):
  h = _headers(token)
  h["Content-Type"] = content_type
  req = urllib.request.Request(url, data=data, method=method, headers=h)
  with urllib.request.urlopen(req) as r:
    return r.read()


def get_or_create_release(tag: str, title: str, body: str, prerelease: bool) -> dict:
  """Get a release by tag, or create it if missing.

  Args:
    tag: Release tag (e.g. 'te-rocm-wheels').
    title: Release title.
    body: Release body text.
    prerelease: Whether to mark the release as a prerelease.

  Returns:
    The GitHub release object JSON.
  """
  token = _env_token()
  owner, name = _env_repo()
  try:
    rel = _request_json("GET", f"{API}/repos/{owner}/{name}/releases/tags/{tag}", token)
    if rel:
      return rel
  except urllib.error.HTTPError as e:
    if e.code != 404:
      raise
  return _request_json(
      "POST",
      f"{API}/repos/{owner}/{name}/releases",
      token,
      {"tag_name": tag, "name": title, "body": body, "prerelease": prerelease},
  )


def upload_asset(tag: str, title: str, body: str, file_path: str, prerelease: bool) -> None:
  """Upload (replace) a release asset under the given tag.

  If an asset with the same filename already exists, it is deleted first.
  """
  token = _env_token()
  owner, name = _env_repo()

  rel = get_or_create_release(tag, title, body, prerelease)
  release_id = rel["id"]
  upload_url = rel["upload_url"].split("{", 1)[0]

  assets = _request_json("GET", f"{API}/repos/{owner}/{name}/releases/{release_id}", token)["assets"]
  file_name = os.path.basename(file_path)
  for a in assets:
    if a.get("name") == file_name:
      _request_json("DELETE", f"{API}/repos/{owner}/{name}/releases/assets/{a['id']}", token)

  with open(file_path, "rb") as f:
    data = f.read()
  up = f"{upload_url}?{urllib.parse.urlencode({'name': file_name})}"
  _request_raw("POST", up, token, data, "application/octet-stream")
  print(f"Uploaded {file_name} to release tag {tag}", flush=True)


def prune_assets(tag: str, keep_days: int) -> None:
  """Delete assets older than `keep_days` from the given release tag."""
  token = _env_token()
  owner, name = _env_repo()
  try:
    rel = _request_json("GET", f"{API}/repos/{owner}/{name}/releases/tags/{tag}", token)
  except urllib.error.HTTPError as e:
    if e.code == 404:
      print(f"No release for tag {tag}; skipping asset prune.", flush=True)
      return
    raise

  cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)
  pruned = 0
  for a in rel.get("assets", []):
    created = dt.datetime.fromisoformat(a["created_at"].replace("Z", "+00:00"))
    if created < cutoff:
      _request_json("DELETE", f"{API}/repos/{owner}/{name}/releases/assets/{a['id']}", token)
      pruned += 1
      print(f"Pruned old asset: {a['name']} (created_at={a['created_at']})", flush=True)
  print(f"Asset prune complete for {tag}. Deleted {pruned} assets older than {keep_days} days.", flush=True)


def prune_releases(prefix: str, keep_days: int) -> None:
  """Delete releases with tag names starting with `prefix` older than `keep_days` days."""
  token = _env_token()
  owner, name = _env_repo()
  cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=keep_days)

  page = 1
  deleted = 0
  while True:
    rels = _request_json("GET", f"{API}/repos/{owner}/{name}/releases?per_page=100&page={page}", token)
    if not rels:
      break
    for rel in rels:
      tag = rel.get("tag_name", "")
      if not tag.startswith(prefix):
        continue
      created = dt.datetime.fromisoformat(rel["created_at"].replace("Z", "+00:00"))
      if created < cutoff:
        _request_json("DELETE", f"{API}/repos/{owner}/{name}/releases/{rel['id']}", token)
        deleted += 1
        print(f"Deleted old release {tag} (created_at={rel['created_at']})", flush=True)
    page += 1
  print(f"Release prune complete. Deleted {deleted} releases older than {keep_days} days.", flush=True)


def main(argv: list[str]) -> int:
  p = argparse.ArgumentParser()
  sub = p.add_subparsers(dest="cmd", required=True)

  up = sub.add_parser("upload")
  up.add_argument("--tag", required=True)
  up.add_argument("--title", required=True)
  up.add_argument("--body", required=True)
  up.add_argument("--file", required=True)
  prg = up.add_mutually_exclusive_group()
  prg.add_argument("--prerelease", dest="prerelease", action="store_true")
  prg.add_argument("--no-prerelease", dest="prerelease", action="store_false")
  up.set_defaults(prerelease=True)

  pa = sub.add_parser("prune-assets")
  pa.add_argument("--tag", required=True)
  pa.add_argument("--keep-days", type=int, required=True)

  pr = sub.add_parser("prune-releases")
  pr.add_argument("--prefix", required=True)
  pr.add_argument("--keep-days", type=int, required=True)

  args = p.parse_args(argv)

  if args.cmd == "upload":
    upload_asset(args.tag, args.title, args.body, args.file, prerelease=args.prerelease)
    return 0
  if args.cmd == "prune-assets":
    prune_assets(args.tag, args.keep_days)
    return 0
  if args.cmd == "prune-releases":
    prune_releases(args.prefix, args.keep_days)
    return 0
  raise AssertionError("unreachable")


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
