# Copyright 2026
# Licensed under the Apache License, Version 2.0.
"""Exact-batch REPLAY input pipeline (dataset_type=megatron_replay).

Replays a directory of pre-dumped Megatron/Primus per-step global batches
VERBATIM, in order, bypassing every data-pipeline difference (sample
construction, document handling, shuffle RNG). This isolates a MaxText
data-pipeline mismatch from an optimizer / update-path mismatch: feed MaxText
the IDENTICAL batches Megatron trained on and overlay per-step loss.

The dump is produced by ``mega_analysis/code/dump_megatron_batches.py`` and
contains, with shape ``(N, GBS, S)``:
  input_ids.npy   int32   GPTDataset 'tokens'  (text[:-1])
  labels.npy      int32   GPTDataset 'labels'  (text[1:])
  loss_mask.npy   uint8   GPTDataset 'loss_mask' (1 = scored token)
  position_ids.npy (S,) int32  single-segment arange
  meta.json

MaxText weights the loss by ``(targets_segmentation != 0)`` (train.py), so we
map Megatron's per-token ``loss_mask`` directly onto the segmentation field:
this reproduces Megatron's exact loss normalization (sum over scored tokens).
For the MLPerf megatron .bin path loss_mask is all-ones (every token scored),
so segmentation is all-ones => standard full causal attention.
"""
import json
import os

import jax
import numpy as np
import tensorflow as tf

from maxtext.input_pipeline import multihost_dataloading
from maxtext.utils import max_logging


def _load_meta(replay_dir):
  with open(os.path.join(replay_dir, "meta.json"), "r") as f:
    return json.load(f)


def _make_dataset(replay_dir, seq_len, global_batch_to_load, num_epoch=1):
  input_ids = np.load(os.path.join(replay_dir, "input_ids.npy"), mmap_mode="r")
  labels = np.load(os.path.join(replay_dir, "labels.npy"), mmap_mode="r")
  loss_mask = np.load(os.path.join(replay_dir, "loss_mask.npy"), mmap_mode="r")
  n_steps, gbs, s = input_ids.shape
  assert s == seq_len, f"dump seq_len {s} != config max_target_length {seq_len}"
  meta = _load_meta(replay_dir)
  max_logging.log(
      f"[megatron_replay] {replay_dir}: N={n_steps} GBS={gbs} S={s} "
      f"(meta hash={meta.get('unique_description_hash')}, "
      f"loss_mask_all_ones={meta.get('loss_mask_all_ones')})"
  )

  def gen():
    for _ in range(max(1, num_epoch)):
      for step in range(n_steps):
        for b in range(gbs):
          yield (
              np.asarray(input_ids[step, b], dtype=np.int32),
              np.asarray(labels[step, b], dtype=np.int32),
              np.asarray(loss_mask[step, b], dtype=np.int32),
          )

  output_signature = (
      tf.TensorSpec(shape=[seq_len], dtype=tf.int32),
      tf.TensorSpec(shape=[seq_len], dtype=tf.int32),
      tf.TensorSpec(shape=[seq_len], dtype=tf.int32),
  )
  ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

  pos = tf.range(seq_len, dtype=tf.int32)

  def _fmt(inputs, targets, lmask):
    # Megatron loss_mask -> segmentation so MaxText's (segmentation != 0) loss
    # weighting reproduces Megatron's exact token normalization.
    seg = lmask
    return {
        "inputs": inputs,
        "targets": targets,
        "inputs_segmentation": seg,
        "targets_segmentation": seg,
        "inputs_position": pos,
        "targets_position": pos,
    }

  ds = ds.map(_fmt, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(global_batch_to_load // jax.process_count(), drop_remainder=True)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def make_megatron_replay_train_iterator(config, global_mesh, process_indices):
  assert config.megatron_replay_path, "megatron_replay requires megatron_replay_path"
  ds = _make_dataset(
      config.megatron_replay_path,
      config.max_target_length,
      config.global_batch_size_to_load,
      num_epoch=max(1, config.num_epoch),
  )
  return multihost_dataloading.MultiHostDataLoadIterator(ds, global_mesh)


def make_megatron_replay_eval_iterator(config, global_mesh, process_indices):
  # The exact-batch experiment compares per-step TRAIN loss; eval is normally
  # disabled (eval_interval=0). If enabled, replay the same batches.
  ds = _make_dataset(
      config.megatron_replay_path,
      config.max_target_length,
      config.global_batch_size_to_load_eval,
      num_epoch=1,
  )
  return multihost_dataloading.MultiHostDataLoadIterator(ds, global_mesh)
