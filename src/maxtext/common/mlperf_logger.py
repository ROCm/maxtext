# Copyright 2026 Google LLC
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

"""Optional MLPerf-compliance MLLOG emitter for llama3.1-8b pretraining.

Default-OFF (gated by ``config.enable_mlperf_logging``). When enabled, emits the
MLPerf Training v6.0 ``llama31_8b`` Closed key set via ``mlperf_logging.mllog`` so
a MaxText run can be validated with ``mlperf_logging.compliance_checker`` and
``rcp_checker``. Mirrors the reference emitter
(``training/small_llm_pretraining/nemo/callbacks.py``) adapted to MaxText's
metric-logger call sites.

No-op unless enabled AND ``mlperf_logging`` is importable, and only emits on
``jax.process_index() == 0``. The emitted HP values are read straight from the
resolved config, so the log is TRUTHFUL: it only passes the checker when the run
is configured compliantly (e.g. ``learning_rate_schedule_steps == 1_200_000`` so
``decay_steps == 1_200_000 - warmup_steps`` and ``max_steps == 1_200_000``).

Key/value contract (training_6.0.0/closed_llama31_8b.yaml + common.yaml):
  cache_clear -> init_start -> submission_* + HP config events -> init_stop ->
  run_start -> block_start -> [eval_start, eval_stop, eval_accuracy]* ->
  block_stop -> run_stop(status) + train_samples + eval_samples.
"""
import os

import jax

try:
  from mlperf_logging import mllog
  from mlperf_logging.mllog import constants as _c

  _MLLOG_AVAILABLE = True
except Exception:  # pragma: no cover - mlperf_logging optional
  _MLLOG_AVAILABLE = False


class MLPerfLogger:
  """Thin, default-off MLLOG emitter driven by the MaxText MetricLogger."""

  def __init__(self, config):
    self.config = config
    self.enabled = (
        bool(getattr(config, "enable_mlperf_logging", False))
        and _MLLOG_AVAILABLE
        and jax.process_index() == 0
    )
    self._run_started = False
    self._run_stopped = False
    self._last_consumed_samples = 0
    if not self.enabled:
      return

    path = getattr(config, "mlperf_log_path", "") or os.path.join(
        (getattr(config, "base_output_directory", "") or "."),
        (getattr(config, "run_name", "") or "run"),
        "mlperf.log",
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    self._logger = mllog.get_mllogger()
    mllog.config(default_stack_offset=2, filename=path)

  # --- helpers ---------------------------------------------------------------
  def _gbs(self):
    return int(self.config.global_batch_size_to_train_on)

  def _consumed(self, train_step):
    # Samples consumed after completing (0-indexed) train_step.
    return int((train_step + 1) * self._gbs())

  def _event(self, key, value=None, metadata=None):
    self._logger.event(key=key, value=value, metadata=metadata or {})

  def _start(self, key, value=None, metadata=None):
    self._logger.start(key=key, value=value, metadata=metadata or {})

  def _end(self, key, value=None, metadata=None):
    self._logger.end(key=key, value=value, metadata=metadata or {})

  # --- lifecycle -------------------------------------------------------------
  def log_init(self):
    """cache_clear + init_start + submission info + all HP config events."""
    if not self.enabled:
      return
    cfg = self.config
    warmup_steps = int(round(cfg.warmup_steps_fraction * cfg.learning_rate_schedule_steps))
    decay_steps = int(cfg.learning_rate_schedule_steps) - warmup_steps
    end_lr = float(cfg.learning_rate) * float(cfg.learning_rate_final_fraction)
    eval_samples = int(cfg.eval_steps) * self._gbs() if int(cfg.eval_steps) > 0 else 0

    self._event(_c.CACHE_CLEAR, value=True)
    self._start(_c.INIT_START)
    # submission info
    self._event(_c.SUBMISSION_ORG, value=cfg.mlperf_submission_org)
    self._event(_c.SUBMISSION_PLATFORM, value=cfg.mlperf_submission_platform)
    self._event(_c.SUBMISSION_DIVISION, value=cfg.mlperf_submission_division)
    self._event(_c.SUBMISSION_BENCHMARK, value=cfg.mlperf_submission_benchmark)
    self._event(_c.SUBMISSION_STATUS, value=_c.ONPREM)
    # HP / config events required by closed_llama31_8b.yaml
    self._event(_c.GLOBAL_BATCH_SIZE, value=self._gbs())
    self._event(_c.MAX_SEQUENCE_LENGTH, value=int(cfg.max_target_length))
    self._event(_c.OPT_NAME, value="adamw")
    self._event(_c.OPT_BASE_LR, value=float(cfg.learning_rate))
    self._event(_c.OPT_END_LR, value=end_lr)
    self._event(_c.OPT_LR_WARMUP_STEPS, value=warmup_steps)
    self._event(_c.OPT_LR_DECAY_STEPS, value=decay_steps)
    self._event(_c.OPT_LR_DECAY_SCHEDULE, value="cosine with linear warmup")
    self._event(_c.OPT_ADAMW_BETA_1, value=float(cfg.adam_b1))
    self._event(_c.OPT_ADAMW_BETA_2, value=float(cfg.adam_b2))
    self._event(_c.OPT_ADAMW_EPSILON, value=float(cfg.adam_eps))
    self._event(_c.OPT_ADAMW_WEIGHT_DECAY, value=float(cfg.adam_weight_decay))
    self._event(_c.OPT_GRADIENT_CLIP_NORM, value=float(cfg.gradient_clipping_threshold))
    self._event(_c.GRADIENT_ACCUMULATION_STEPS, value=int(cfg.gradient_accumulation_steps))
    self._event(_c.EVAL_SAMPLES, value=eval_samples)
    self._event("max_steps", value=int(cfg.learning_rate_schedule_steps))
    if cfg.mlperf_lowest_precision_linear:
      self._event("lowest_numerical_precision_in_linear", value=cfg.mlperf_lowest_precision_linear)

  def maybe_start_run(self):
    """init_stop + run_start + block_start (emitted once, before the first eval)."""
    if not self.enabled or self._run_started:
      return
    self._end(_c.INIT_STOP)
    self._start(_c.RUN_START)
    self._start(_c.BLOCK_START, metadata={_c.SAMPLES_COUNT: 0})
    self._run_started = True

  def log_eval(self, train_step, eval_loss):
    """eval_start + eval_stop + eval_accuracy(value=log-ppl, samples_count)."""
    if not self.enabled:
      return
    self.maybe_start_run()
    samples = self._consumed(train_step)
    self._last_consumed_samples = samples
    self._start(_c.EVAL_START, metadata={_c.SAMPLES_COUNT: samples})
    self._end(_c.EVAL_STOP, metadata={_c.SAMPLES_COUNT: samples})
    self._event(_c.EVAL_ACCURACY, value=float(eval_loss), metadata={_c.SAMPLES_COUNT: samples})

  def log_run_stop(self, success, train_step=None):
    """block_stop + run_stop(status) + train_samples (emitted once).

    eval_samples is emitted once in ``log_init`` (it is a config HP, EXACTLY_ONE),
    so it is NOT re-emitted here.
    """
    if not self.enabled or self._run_stopped:
      return
    self.maybe_start_run()
    samples = self._consumed(train_step) if train_step is not None else self._last_consumed_samples
    self._end(_c.BLOCK_STOP, metadata={_c.SAMPLES_COUNT: samples})
    self._end(_c.RUN_STOP, metadata={"status": _c.SUCCESS if success else _c.ABORTED})
    self._event("train_samples", value=samples)
    self._run_stopped = True
