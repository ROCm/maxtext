# Copyright 2026
# Licensed under the Apache License, Version 2.0.
"""Input pipeline for MLPerf Megatron-indexed (.bin/.idx) preprocessed C4.

The MLPerf llama3.1-8b preprocessed dataset is a Megatron `MMapIndexedDataset`:
a flat concatenation of int32 token ids (the `.bin`) over documents (boundaries
in the `.idx`). Standard GPT pretraining builds samples by chunking the
concatenated token stream into fixed `max_target_length`-token blocks with FULL
causal attention across document boundaries (no per-doc reset). We replicate that:

  sample i = bin[i*S : i*S + S + 1]  (stride S = max_target_length; read S+1 for
             the next-token shift), inputs = sample[:-1], targets = sample[1:].

Train: shuffle the sample order (seeded). Eval: MLPerf protocol = first 1024
chunks of the validation stream (~8.4M tokens), no shuffle.

The .bin paths are passed via config.grain_train_files / config.grain_eval_files
(reused as generic path strings). Tokens are pre-tokenized (the MLPerf Llama-3.1
tokenizer), so there is NO tokenization at load time (cheap memmap slicing).
"""
import numpy as np
import tensorflow as tf
import jax

from maxtext.input_pipeline import multihost_dataloading

EVAL_NUM_SEQUENCES = 1024  # MLPerf eval protocol: first 1024 sequences (~8.4M tokens)


def _make_dataset(bin_path, seq_len, global_batch_to_load, shuffle, seed,
                  num_epoch=1, max_samples=None):
  arr = np.memmap(bin_path, dtype=np.int32, mode="r")
  n_samples = (arr.shape[0] - 1) // seq_len
  if max_samples is not None:
    n_samples = min(n_samples, max_samples)

  def gen():
    order = np.arange(n_samples)
    for ep in range(num_epoch):
      if shuffle:
        np.random.default_rng(seed + ep).shuffle(order)
      for i in order:
        s = int(i) * seq_len
        yield np.asarray(arr[s:s + seq_len + 1], dtype=np.int32)

  ds = tf.data.Dataset.from_generator(
      gen, output_signature=tf.TensorSpec(shape=[seq_len + 1], dtype=tf.int32))

  def _fmt(x):
    inputs = x[:-1]
    targets = x[1:]
    seg = tf.ones([seq_len], dtype=tf.int32)        # full causal: one segment per row
    pos = tf.range(seq_len, dtype=tf.int32)
    return {
        "inputs": inputs, "targets": targets,
        "inputs_segmentation": seg, "targets_segmentation": seg,
        "inputs_position": pos, "targets_position": pos,
    }

  ds = ds.map(_fmt, num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.batch(global_batch_to_load // jax.process_count(), drop_remainder=True)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def make_megatron_train_iterator(config, global_mesh, process_indices):
  ds = _make_dataset(
      config.grain_train_files, config.max_target_length,
      config.global_batch_size_to_load, shuffle=config.enable_data_shuffling,
      seed=config.data_shuffle_seed, num_epoch=max(1, config.num_epoch))
  return multihost_dataloading.MultiHostDataLoadIterator(ds, global_mesh)


def make_megatron_eval_iterator(config, global_mesh, process_indices):
  ds = _make_dataset(
      config.grain_eval_files, config.max_target_length,
      config.global_batch_size_to_load_eval, shuffle=False,
      seed=config.data_shuffle_seed, num_epoch=1, max_samples=EVAL_NUM_SEQUENCES)
  return multihost_dataloading.MultiHostDataLoadIterator(ds, global_mesh)
