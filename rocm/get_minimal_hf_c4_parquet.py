#!/usr/bin/env python3
"""Generate minimal HuggingFace-style C4 parquet shards using the MinIO instance.

Same logic with ArrayRecord minimal generator (range selection + smallest shard) but for parquet.

Outputs:
  rocm/c4_en_dataset_minimal/hf/c4/c4-train-00000-of-01637.parquet
  rocm/c4_en_dataset_minimal/hf/c4/c4-validation-00000-of-01637.parquet

Appends logs to: rocm/gcloud_decoupled_test_logs/minimal_hf_parquet.log
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from minio import Minio
from minio.error import S3Error
import pyarrow as pa
import pyarrow.parquet as pq

# ------------ Remote Source Config (override via env) ------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio-frameworks.amd.com")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "hidden")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "hidden")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "true").lower() == "true"
BUCKET = os.environ.get("MINIO_C4_BUCKET", "datasets.dl")

TRAIN_PREFIX = "c4/en/3.0.1/c4-train-"
VAL_PREFIX = "c4/en/3.0.1/c4-validation-"

# ------------ Local Output Paths ------------
BASE_DIR = Path(__file__).parent / "c4_en_dataset_minimal" / "hf" / "c4"
TRAIN_FILE = BASE_DIR / "c4-train-00000-of-01637.parquet"
VAL_FILE = BASE_DIR / "c4-validation-00000-of-01637.parquet"

LOG_DIR = Path(__file__).parent / "gcloud_decoupled_test_logs"
LOG_FILE = LOG_DIR / "minimal_hf_parquet.log"


def log(msg: str):
  LOG_DIR.mkdir(parents=True, exist_ok=True)
  with LOG_FILE.open("a", encoding="utf-8") as f:
    f.write(msg + "\n")
  print(msg)


def list_smallest(client: Minio, prefix: str):
  objs = sorted(
      (o for o in client.list_objects(BUCKET, prefix=prefix, recursive=False)),
      key=lambda o: getattr(o, "size", float("inf")),
  )
  return objs[0] if objs else None


def fetch_rows(client: Minio, obj, row_cap: int) -> List[str]:
  """Download parquet object and extract up to row_cap rows of 'text'."""
  data = client.get_object(BUCKET, obj.object_name)
  try:
    blob = data.read()
  finally:
    data.close(); data.release_conn()
  table = pq.read_table(pa.BufferReader(blob))
  # Heuristic: if no 'text' column, pick first string column.
  col_name = "text" if "text" in table.column_names else table.column_names[0]
  col = table[col_name]
  rows = [str(col[i].as_py()) for i in range(min(row_cap, col.length()))]
  # Normalize minimal text (trim) to reduce size variability.
  rows = [r.strip() for r in rows if isinstance(r, str) and r.strip()]
  return rows


def write_parquet(path: Path, rows: List[str], force: bool):
  if path.exists() and not force:
    log(f"[skip] {path} exists")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  table = pa.Table.from_pydict({"text": rows})
  pq.write_table(table, path, compression="ZSTD")
  log(f"[write] {path} rows={len(rows)} size_kib={path.stat().st_size/1024:.1f}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--force", action="store_true", help="Overwrite existing files")
  parser.add_argument("--train-rows", type=int, default=800, help="Max train rows to sample")
  parser.add_argument("--val-rows", type=int, default=160, help="Max validation rows to sample")
  args = parser.parse_args()

  client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
  if not client.bucket_exists(BUCKET):
    log(f"Bucket '{BUCKET}' does not exist. Abort.")
    return

  try:
    train_obj = list_smallest(client, TRAIN_PREFIX)
    val_obj = list_smallest(client, VAL_PREFIX)
  except S3Error as e:
    log(f"List error: {e}")
    return

  if not train_obj or not val_obj:
    log("Missing smallest shard(s); abort.")
    return

  log(f"Train shard: {train_obj.object_name} size={train_obj.size/1024/1024:.2f} MiB")
  log(f"Val shard: {val_obj.object_name} size={val_obj.size/1024/1024:.2f} MiB")

  try:
    train_rows = fetch_rows(client, train_obj, args.train_rows)
    val_rows = fetch_rows(client, val_obj, args.val_rows)
  except S3Error as e:
    log(f"Download error: {e}")
    return

  write_parquet(TRAIN_FILE, train_rows, args.force)
  write_parquet(VAL_FILE, val_rows, args.force)
  log("Done. Minimal HF parquet dataset generated.")


if __name__ == "__main__":
  main()
