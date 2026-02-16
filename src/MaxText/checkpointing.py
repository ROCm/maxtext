# Copyright 2023–2025 Google LLC
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

"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

import base64
import ctypes
import gc
import time
from typing import Any, Optional

from absl import flags
from etils import epath
from flax.training import train_state
import jax
from MaxText import exceptions
from MaxText import max_logging
from MaxText.globals import DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
from MaxText.multihost_dataloading import MultiHostDataLoadIterator
from MaxText.input_pipeline.input_pipeline_interface import PlaceHolderDataIterator
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.arrays import sharding as sharding_utils
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager
import orbax.checkpoint._src.multihost.multislice as _orbax_multislice
import dataclasses
import json
# pylint: disable=too-many-positional-arguments

import grain
from grain.python import PyGrainCheckpointHandler

CheckpointManager = ocp.CheckpointManager
CheckpointManagerOptions = ocp.CheckpointManagerOptions
Composite = ocp.args.Composite
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
EmergencyCheckpointManager = emergency_checkpoint_manager.CheckpointManager
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = emergency_checkpoint_manager.PersistentCheckpointOptions
EmergencyReplicatorCheckpointManager = emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager
_original_broadcast_one_replica_to_all = _orbax_multislice.broadcast_one_replica_to_all


class GrainCheckpointHandler(PyGrainCheckpointHandler, ocp.CheckpointHandler):
  """A CheckpointHandler that allows specifying process_index and process_count."""

  def save(
      self,
      directory: epath.Path,
      # `item` is for backwards compatibility with older Orbax API, see
      # https://orbax.readthedocs.io/en/latest/api_refactor.html.
      item: Optional[Any] = None,
      args: Any = None,
  ):
    """Saves the given iterator to the checkpoint in `directory`."""
    item = item or args.item  # pytype:disable=attribute-error

    def save_single_process(item, process_index, process_count):
      filename = directory / f"process_{process_index}-of-{process_count}.json"
      if isinstance(item, grain.DatasetIterator):
        state = json.dumps(item.get_state(), indent=4)
      else:
        state = item.get_state().decode()
      filename.write_text(state)

    if isinstance(item, list):
      for local_iterator, process_index, process_count in item:
        save_single_process(local_iterator, process_index, process_count)
    else:
      process_index, process_count = jax.process_index(), jax.process_count()
      save_single_process(item, process_index, process_count)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[Any] = None,
      args: Any = None,
  ) -> Any:
    """Restores the given iterator from the checkpoint in `directory`."""
    item = item or args.item
    process_index = getattr(args, "process_index", None)
    process_count = getattr(args, "process_count", None)

    def restore_single_process(item, process_index, process_count):
      filename = directory / f"process_{process_index}-of-{process_count}.json"
      if not filename.exists():
        raise ValueError(f"File {filename} does not exist.")
      state = filename.read_text()
      if isinstance(item, grain.DatasetIterator):
        state = json.loads(state)
      else:
        state = state.encode()
      item.set_state(state)
      return item

    if isinstance(item, list):
      restored_items = []
      for data_iter, process_idx in zip(item, process_index):
        restored_items.append(restore_single_process(data_iter, process_idx, process_count))
      return restored_items
    else:
      if process_index is None or process_count is None:
        process_index, process_count = jax.process_index(), jax.process_count()
      return restore_single_process(item, process_index, process_count)


@ocp.args.register_with_handler(GrainCheckpointHandler, for_save=True)
@dataclasses.dataclass
class GrainCheckpointSave(ocp.args.CheckpointArgs):
  item: Any


@ocp.args.register_with_handler(GrainCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class GrainCheckpointRestore(ocp.args.CheckpointArgs):
  item: Any
  process_index: Optional[int | list[int]] = None
  process_count: Optional[int] = None


def _load_full_state_from_path(
    path,
    abstract_unboxed_pre_state,
    enable_orbax_v1,
    checkpoint_conversion_fn,
    source_checkpoint_layout,
):
  """Load full state from checkpoint at specified path.

  Args:
    path: path to checkpoint
    abstract_unboxed_pre_state: an abstract state that Orbax matches type
      against.
    enable_orbax_v1: whether to use orbax v1 or the previously supported v0.
    checkpoint_conversion_fn: user-provided function to convert checkpoint to
      maxtext-supported state.
    source_checkpoint_layout: String representation of the checkpoint layout of
      the source checkpoint.

  Returns:
    The loaded state.
  """

  if enable_orbax_v1:
    if source_checkpoint_layout == "orbax":
      context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.ORBAX)
      with context:
        return ocp_v1.load_pytree(path, abstract_unboxed_pre_state)
    elif source_checkpoint_layout == "safetensors":
      context = ocp_v1.Context(checkpoint_layout=ocp_v1.options.CheckpointLayout.SAFETENSORS)
      with context:
        metadata = ocp_v1.pytree_metadata(path)
        simple_abstract_state = metadata.metadata
        shardings = sharding_utils.construct_maximal_shardings(simple_abstract_state)

        def combine_sharding(sds, shardings):
          return jax.ShapeDtypeStruct(shape=sds.shape, dtype=sds.dtype, sharding=shardings)

        sharded_abstract_state = jax.tree.map(combine_sharding, simple_abstract_state, shardings)
        pre_transformed_state = ocp_v1.load_pytree(path, sharded_abstract_state)
      state = checkpoint_conversion_fn(pre_transformed_state)
      return state
    else:
      raise ocp_v1.errors.InvalidLayoutError(f"Unknown checkpoint layout: {source_checkpoint_layout}")
  else:
    # Original v0 logic.
    p = epath.Path(path)
    return ocp.StandardCheckpointer().restore(p, abstract_unboxed_pre_state)


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: None | str = "tfds",
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    max_to_keep: int = 5,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None

  max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

  # Base configuration for all dataset types
  item_names = ("items",)
  # we need to use ocdbt and zarr3 to control max file size in the checkpoint
  item_handlers = {"items": PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)}

  if dataset_type == "grain":
    item_names += ("iter",)
    item_handlers["iter"] = GrainCheckpointHandler()

  # local storage checkpoint needs parent directory created
  p = epath.Path(checkpoint_dir)
  p.mkdir(exist_ok=True, parents=True)
  manager = CheckpointManager(
      p,
      item_names=item_names,
      item_handlers=item_handlers,
      options=CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
          max_to_keep = max_to_keep,
      ),
      logger=orbax_logger,
  )

  max_logging.log("Checkpoint manager created!")
  return manager


def create_orbax_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: Any,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
    orbax_logger: Any = None,  # pytype: disable=attribute-error
):
  """Returns an emergency checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency checkpoint manager...")

  # Only create directories if running on GPUs as the previous
  # directory structure might be assumed by TPUs
  if global_mesh.devices.flatten()[0].platform == "gpu":
    # pylint: disable=protected-access
    local_checkpoint_dir = f"{local_checkpoint_dir}/{jax._src.distributed.global_state.process_id}"
    local_p = epath.Path(local_checkpoint_dir)
    persistent_p = epath.Path(persistent_checkpoint_dir)
    local_p.mkdir(exist_ok=True, parents=True)
    persistent_p.mkdir(exist_ok=True, parents=True)

  manager = EmergencyCheckpointManager(
      local_checkpoint_dir,
      epath.Path(persistent_checkpoint_dir),
      global_mesh=global_mesh,
      abstract_state=abstract_state,
      options=emergency_checkpoint_manager.CheckpointManagerOptions(
          local=LocalCheckpointOptions(save_interval_steps=local_save_interval_steps),
          persistent=PersistentCheckpointOptions(save_interval_steps=persistent_save_interval_steps),
      ),
      logger=orbax_logger,
  )

  max_logging.log("Emergency checkpoint manager created!")
  return manager


def create_orbax_emergency_replicator_checkpoint_manager(
    local_checkpoint_dir: str,
    save_interval_steps: int,
    global_mesh: jax.sharding.Mesh,
):
  """Returns an emergency replicator checkpoint manager."""
  flags.FLAGS.experimental_orbax_use_distributed_process_id = True
  max_logging.log("Creating emergency replicator checkpoint manager...")

  manager = EmergencyReplicatorCheckpointManager(
      epath.Path(local_checkpoint_dir),
      options=emergency_replicator_checkpoint_manager.ReplicatorCheckpointManagerOptions(
          save_interval_steps=save_interval_steps,
      ),
      global_mesh=global_mesh,
  )

  max_logging.log("Emergency replicator checkpoint manager created!")
  return manager


def replicator_error_handler(config: Any):
  """Replicator error handler to handle errors in replicator service."""
  if config.enable_multi_tier_checkpointing:
    local_dir = config.local_checkpoint_directory
    replicator_errors_file = f"{local_dir}/replicator.errors"
    replicator_failed_file = f"{local_dir}/replicator.failed"
    process_replicator_error_file(replicator_errors_file)

    # if the replicator.failed file exists, then we have a fatal error
    is_fatal = process_replicator_error_file(replicator_failed_file)
    if is_fatal:
      raise ValueError("Replicator fatal error found in replicator.failed file.")


def process_replicator_error_file(error_file: str) -> bool:
  """Handles replicator errors by reading, logging, cleaning the error file."""
  error_file_path_exists = epath.Path(error_file).exists()
  if error_file_path_exists:
    max_logging.log(f"replicator_error_handler: file found: {error_file}.")
    read_replicator_error_file(error_file)
    cleanup_replicator_error_file(error_file)

  return error_file_path_exists


def read_replicator_error_file(error_file: str):
  """Read replicator errors file."""
  try:
    error_data = epath.Path(error_file).read_text()
    max_logging.log(f"Contents of replicator error file:\n{error_data}")
  except (OSError, ValueError) as e:
    max_logging.log("replicator_error_handler: Failed to read contents of failed" f" file: {e}")


def cleanup_replicator_error_file(error_file: str):
  """Clean up replicator errors file."""
  try:
    epath.Path(error_file).unlink()
  except (OSError, ValueError) as e:
    max_logging.log("replicator_error_handler: Failed to remove replicator errors file:" f" {e}")


def print_save_message(step, async_checkpointing):
  if async_checkpointing:
    max_logging.log(f"Started an asynchronous checkpoint save for step {step}")
  else:
    max_logging.log(f"Saved a checkpoint at step {step}.")


# ── Custom broadcast (avoids RCCL communicator leak) ────────────────────────
#
# Orbax's broadcast_one_replica_to_all creates RCCL communicators cached in
# XLA's GPU clique system with persistent proxy threads that cannot be released
# from Python, degrading training TGS.
#
# PRIMARY: Direct RCCL via ctypes (ncclCommInitRank / ncclBroadcast /
#   ncclCommDestroy + gc.collect()).  Uses the RDMA backend network; no CPU
#   copies.  Communicators are destroyed and Python references cleared
#   immediately so proxy threads do not survive into the training loop.
# FALLBACK: Orbax default broadcast (leaks threads but is correct/portable).
# ────────────────────────────────────────────────────────────────────────────

_NCCL_UNIQUE_ID_BYTES = 128
_NCCL_INT8 = 0  # ncclInt8: treat all data as raw bytes
_BROADCAST_BARRIER_TIMEOUT_MS = 1_800_000  # 30 min


class _NcclUniqueId(ctypes.Structure):
  # c_ubyte (not c_char) avoids null-terminated string semantics: bytes()
  # returns all 128 bytes and ctypes.memmove targets the struct's memory.
  _fields_ = [("internal", ctypes.c_ubyte * _NCCL_UNIQUE_ID_BYTES)]


class _NativeLibsNotFoundError(RuntimeError):
  """Raised when RCCL/NCCL or HIP/CUDA runtime libraries are unavailable."""


# Lazy-loaded native library handles (populated by _get_native_libs).
_cached_nccl_lib = None
_cached_gpu_rt_ops = None  # (set_device, stream_create, stream_sync, stream_destroy)


def _load_shared_lib(candidates):
  """Try loading the first available shared library from *candidates*."""
  for name in candidates:
    try:
      return ctypes.CDLL(name)
    except OSError:
      continue
  return None


def _setup_nccl_bindings(lib):
  """Set argtypes/restype for the NCCL/RCCL functions we call."""
  lib.ncclGetUniqueId.restype = ctypes.c_int
  lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]

  lib.ncclCommInitRank.restype = ctypes.c_int
  lib.ncclCommInitRank.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),  # ncclComm_t* comm
      ctypes.c_int,                      # int nranks
      _NcclUniqueId,                     # ncclUniqueId commId (by value)
      ctypes.c_int,                      # int rank
  ]

  lib.ncclBroadcast.restype = ctypes.c_int
  lib.ncclBroadcast.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
      ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
  ]

  for fn_name in ("ncclGroupStart", "ncclGroupEnd"):
    fn = getattr(lib, fn_name)
    fn.restype = ctypes.c_int
    fn.argtypes = []

  lib.ncclCommDestroy.restype = ctypes.c_int
  lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]


def _setup_gpu_runtime_bindings(lib):
  """Set up ctypes bindings for device/stream ops (HIP or CUDA)."""
  prefix = "hip" if hasattr(lib, "hipSetDevice") else "cuda"
  set_device = getattr(lib, f"{prefix}SetDevice")
  stream_create = getattr(lib, f"{prefix}StreamCreate")
  stream_sync = getattr(lib, f"{prefix}StreamSynchronize")
  stream_destroy = getattr(lib, f"{prefix}StreamDestroy")

  set_device.restype = ctypes.c_int
  set_device.argtypes = [ctypes.c_int]
  stream_create.restype = ctypes.c_int
  stream_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
  stream_sync.restype = ctypes.c_int
  stream_sync.argtypes = [ctypes.c_void_p]
  stream_destroy.restype = ctypes.c_int
  stream_destroy.argtypes = [ctypes.c_void_p]

  return set_device, stream_create, stream_sync, stream_destroy


def _get_native_libs():
  """Load and configure NCCL + GPU runtime libs (cached after first call).

  Raises:
    _NativeLibsNotFoundError: if either library is unavailable.
  """
  global _cached_nccl_lib, _cached_gpu_rt_ops
  if _cached_nccl_lib is not None:
    return _cached_nccl_lib, _cached_gpu_rt_ops

  nccl = _load_shared_lib(("librccl.so.1", "librccl.so",
                            "libnccl.so.2", "libnccl.so"))
  gpu_rt = _load_shared_lib(("libamdhip64.so", "libcudart.so"))
  if nccl is None or gpu_rt is None:
    raise _NativeLibsNotFoundError(
        "RCCL/NCCL or HIP/CUDA runtime not found; cannot use direct broadcast")

  _setup_nccl_bindings(nccl)
  _cached_gpu_rt_ops = _setup_gpu_runtime_bindings(gpu_rt)
  _cached_nccl_lib = nccl
  return nccl, _cached_gpu_rt_ops


def _gpu_check(ret, msg="GPU operation"):
  """Raise RuntimeError if an NCCL/HIP/CUDA call returns a non-zero code."""
  if ret != 0:
    raise RuntimeError(f"{msg} failed with error code {ret}")


def _build_rccl_group_mapping(global_mesh, replica_axis_index):
  """Compute per-GPU replica ranks and group IDs for NCCL communicators.

  Each local GPU is assigned:
    - **replica_rank**: its position along the replica axis (0 = source).
    - **group_id**: the device ID of the corresponding source-replica GPU,
      used as a unique key for the NCCL communicator group.

  Returns:
    (dp, num_local, local_ranks, source_dev_ids)
  """
  devices = global_mesh.devices
  dp = devices.shape[replica_axis_index]

  dev_to_pos = {}
  for idx, dev in np.ndenumerate(devices):
    dev_to_pos[dev.id] = idx

  local_devs = sorted(jax.local_devices(), key=lambda d: d.id)
  num_local = len(local_devs)

  local_ranks = []
  source_dev_ids = []
  for dev in local_devs:
    pos = dev_to_pos[dev.id]
    local_ranks.append(pos[replica_axis_index])
    source_pos = list(pos)
    source_pos[replica_axis_index] = 0
    source_dev_ids.append(devices[tuple(source_pos)].id)

  return dp, num_local, local_ranks, source_dev_ids


def _rewrap_as_global_arrays(in_tree, global_mesh):
  """Re-wrap JAX arrays with global NamedSharding after in-place broadcast."""
  out = []
  for arr in in_tree:
    sharding = jax.sharding.NamedSharding(global_mesh, arr.sharding.spec)
    bufs = [s.data for s in sorted(
        arr.addressable_shards, key=lambda s: s.device.id)]
    out.append(jax.make_array_from_single_device_arrays(
        arr.shape, sharding, bufs))
  return out


def _rccl_broadcast_one_replica_to_all(
    in_tree,
    global_mesh,
    replica_axis_index,
    is_source,
    memory_limit_bytes=None,
    memory_scaling_factor=0.75,
):
  """Broadcast via direct RCCL/NCCL calls on the backend RDMA network.

  Creates temporary NCCL communicators, broadcasts GPU data in-place (no CPU
  copies), then destroys the communicators so proxy threads are cleaned up.

  Raises:
    _NativeLibsNotFoundError: if RCCL/HIP libraries are unavailable.
  """
  del memory_limit_bytes, memory_scaling_factor  # match Orbax API signature
  from jax._src import distributed as _jax_distributed

  tree_len = len(in_tree)
  if tree_len == 0:
    return (), 0

  pid = jax.process_index()
  client = _jax_distributed.global_state.client
  kv_prefix = "maxtext_rccl_bcast"

  libnccl, (set_device, stream_create, stream_sync, stream_destroy) = (
      _get_native_libs())

  dp, num_local, local_ranks, source_dev_ids = _build_rccl_group_mapping(
      global_mesh, replica_axis_index)

  # ---- exchange NCCL unique IDs via JAX coordinator -------------------------
  uids = []
  if is_source:
    for g in range(num_local):
      uid = _NcclUniqueId()
      _gpu_check(libnccl.ncclGetUniqueId(ctypes.byref(uid)),
                  "ncclGetUniqueId")
      client.key_value_set(
          f"{kv_prefix}/uid/{source_dev_ids[g]}",
          base64.b64encode(bytes(uid.internal)).decode("ascii"))
      uids.append(uid)

  client.wait_at_barrier(f"{kv_prefix}/uids_published",
                         _BROADCAST_BARRIER_TIMEOUT_MS)

  if not is_source:
    for g in range(num_local):
      uid_b64 = client.blocking_key_value_get(
          f"{kv_prefix}/uid/{source_dev_ids[g]}",
          _BROADCAST_BARRIER_TIMEOUT_MS)
      uid = _NcclUniqueId()
      ctypes.memmove(uid.internal, base64.b64decode(uid_b64),
                      _NCCL_UNIQUE_ID_BYTES)
      uids.append(uid)

  max_logging.log(
      f"RCCL broadcast: host {pid} creating {num_local} communicators "
      f"(dp={dp}, replica_rank={local_ranks[0]})")

  # ---- create communicators, broadcast, then guarantee cleanup --------------
  comms = []
  streams = []
  try:
    # Create NCCL communicators (one per local GPU).
    _gpu_check(libnccl.ncclGroupStart(), "ncclGroupStart (init)")
    for g in range(num_local):
      comm = ctypes.c_void_p()
      set_device(g)
      _gpu_check(
          libnccl.ncclCommInitRank(
              ctypes.byref(comm), dp, uids[g], local_ranks[g]),
          f"ncclCommInitRank GPU {g}")
      comms.append(comm)
    _gpu_check(libnccl.ncclGroupEnd(), "ncclGroupEnd (init)")

    # Create GPU streams.
    for g in range(num_local):
      stream = ctypes.c_void_p()
      set_device(g)
      _gpu_check(stream_create(ctypes.byref(stream)),
                 f"stream create GPU {g}")
      streams.append(stream)

    # Broadcast all parameters in-place.
    t0 = time.monotonic()
    total_bytes = 0

    _gpu_check(libnccl.ncclGroupStart(), "ncclGroupStart (broadcast)")
    for param_idx, arr in enumerate(in_tree):
      for g, shard in enumerate(
          sorted(arr.addressable_shards, key=lambda s: s.device.id)):
        ptr = shard.data.unsafe_buffer_pointer()
        nbytes = shard.data.nbytes
        set_device(g)
        _gpu_check(
            libnccl.ncclBroadcast(
                ctypes.c_void_p(ptr), ctypes.c_void_p(ptr),
                nbytes, _NCCL_INT8, 0, comms[g], streams[g]),
            f"ncclBroadcast param {param_idx} GPU {g}")
        total_bytes += nbytes
    _gpu_check(libnccl.ncclGroupEnd(), "ncclGroupEnd (broadcast)")

    # Synchronize streams.
    for g in range(num_local):
      set_device(g)
      _gpu_check(stream_sync(streams[g]), f"stream sync GPU {g}")

    elapsed = time.monotonic() - t0
    throughput = total_bytes / elapsed / 1e9 if elapsed > 0 else 0
    max_logging.log(
        f"RCCL broadcast: host {pid} transferred {total_bytes / 1e9:.2f} GB "
        f"across {tree_len} params in {elapsed:.1f}s ({throughput:.2f} GB/s)")

  finally:
    # Destroy communicators and streams at the C level, then drop all Python
    # references and force GC.  Without gc.collect(), ctypes c_void_p handles
    # and internal JAX/XLA references can prevent RCCL proxy threads from
    # being torn down until the next non-deterministic GC cycle.
    for comm in comms:
      libnccl.ncclCommDestroy(comm)
    for g, stream in enumerate(streams):
      set_device(g)
      stream_destroy(stream)
    comms.clear()
    streams.clear()
    if num_local:
      gc.collect()
      max_logging.log(
          f"RCCL broadcast: host {pid} destroyed {num_local} communicators "
          "and ran gc.collect()")

  # ---- re-wrap arrays with global sharding ----------------------------------
  out_tree = _rewrap_as_global_arrays(in_tree, global_mesh)

  # ---- barrier & KV cleanup -------------------------------------------------
  client.wait_at_barrier(f"{kv_prefix}/done", _BROADCAST_BARRIER_TIMEOUT_MS)
  if is_source:
    for g in range(num_local):
      try:
        client.key_value_delete(f"{kv_prefix}/uid/{source_dev_ids[g]}")
      except Exception:
        pass
  max_logging.log(f"RCCL broadcast: host {pid} complete")

  return tuple(out_tree), 1


def _custom_broadcast_one_replica_to_all(
    in_tree, global_mesh, replica_axis_index, is_source,
    memory_limit_bytes=None, memory_scaling_factor=0.75,
):
  """Dispatch to direct RCCL broadcast, falling back to Orbax if unavailable.

  The Orbax fallback uses JAX/XLA's clique-cached RCCL communicators which
  leak persistent proxy threads (degrading training TGS), but it is correct
  and portable.
  """
  try:
    return _rccl_broadcast_one_replica_to_all(
        in_tree, global_mesh, replica_axis_index, is_source,
        memory_limit_bytes, memory_scaling_factor)
  except _NativeLibsNotFoundError as e:
    max_logging.log(
        f"RCCL direct broadcast unavailable ({e}), "
        "falling back to Orbax default broadcast")
    return _original_broadcast_one_replica_to_all(
        in_tree, global_mesh, replica_axis_index, is_source,
        memory_limit_bytes, memory_scaling_factor)


def _install_custom_broadcast():
  """Monkeypatch orbax to use direct RCCL broadcast (Orbax fallback)."""
  max_logging.log(
      "Installing custom broadcast (RCCL direct with Orbax fallback) "
      "to avoid RCCL communicator leak")
  _orbax_multislice.broadcast_one_replica_to_all = (
      _custom_broadcast_one_replica_to_all)


def _uninstall_custom_broadcast():
  """Restore orbax's original broadcast function."""
  _orbax_multislice.broadcast_one_replica_to_all = (
      _original_broadcast_one_replica_to_all)
  max_logging.log("Restored original orbax broadcast_one_replica_to_all")


# ── End custom broadcast ────────────────────────────────────────────────────


def _find_idx(array: np.ndarray, replica_axis_idx: int):
  """Returns the index along given dimension that the current host belongs to."""
  idx = None
  for idx, val in np.ndenumerate(array):
    if val.process_index == jax.process_index():
      break
  return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
  """Returns the devices from the replica that current host belongs to.

  Replicas are assumed to be restricted to the first axis.

  Args:
    device_array: devices of the mesh that can be obtained by mesh.devices()
    replica_axis_idx: axis dimension along which replica is taken

  Returns:
    devices inside the replica that current host is in
  """
  idx = _find_idx(device_array, replica_axis_idx)
  replica_result = np.take(device_array, idx, axis=replica_axis_idx)
  return np.expand_dims(replica_result, axis=replica_axis_idx)


def _prepare_scaled_down_grain_restore_args(
    data_iterator: list, process_count_jax: int, process_count_stored: int, directory: epath.Path
) -> GrainCheckpointRestore:
  """
  Prepares the restore arguments for a scaled-up (list) data iterator.

  This is used when restoring a checkpoint saved with more processes than
  the current run (e.g., 64 files onto 32 JAX processes).
  """
  # 1. Validation Assertions
  assert isinstance(data_iterator, list), (
      f"{process_count_stored} processes found in Grain checkpoint directory {directory}, but only "
      f"{process_count_jax} jax processes in this run, please set expansion_factor_real_data accordingly."
  )

  scaling_factor = len(data_iterator)
  expected_process_count = process_count_stored / process_count_jax
  assert scaling_factor == expected_process_count, (
      f"Found {process_count_stored} processes in checkpoint and {process_count_jax} "
      f"JAX processes, implying a scaling factor of {expected_process_count}. "
      f"However, the data_iterator list has {scaling_factor} items."
  )

  # 2. Prepare Arguments
  local_iterator_list = [x.local_iterator for x in data_iterator]
  # Each JAX process calculates the global indices it's responsible for.
  # e.g., process 0 with scaling_factor=2 handles checkpoints from processes [0, 32]
  # e.g., process 1 with scaling_factor=2 handles checkpoints from processes [1, 33]
  process_index_list = [jax.process_index() + i * process_count_jax for i in range(scaling_factor)]

  return GrainCheckpointRestore(local_iterator_list, process_index=process_index_list, process_count=process_count_stored)


def _restore_grain_iterator(
    checkpoint_manager,
    step: int,
    data_iterator,
    checkpoint_args,
    expansion_factor_real_data: int,  # This must be defined in the outer scope
) -> tuple[Any, None]:
  """
  Handles the complex logic for restoring a Grain data iterator checkpoint.
  This function dispatches to the correct restore strategy based on
  the number of stored checkpoint files vs. current JAX processes.
  """
  directory = checkpoint_manager.directory / str(step) / "iter"
  process_count_jax = jax.process_count()

  # Count the number of checkpoint files
  process_count_stored = len(list(directory.glob("process_*-of-*.json")))

  grain_restore_args = None

  if process_count_stored > process_count_jax:
    # Scaling down from a larger number of hosts. (e.g., 128 files -> 64 processes)
    # In this case, each host restores a list of data iterators.
    grain_restore_args = _prepare_scaled_down_grain_restore_args(
        data_iterator, process_count_jax, process_count_stored, directory
    )

  elif process_count_stored == process_count_jax:
    # Normal case: number of hosts is the same. (e.g., 64 files -> 64 processes)
    assert not isinstance(data_iterator, list), (
        f"{process_count_stored} processes found in Grain checkpoint directory {directory}, matching the number of "
        "jax process, please do not set expansion_factor_real_data."
    )
    grain_restore_args = GrainCheckpointRestore(data_iterator.local_iterator)

  elif expansion_factor_real_data > 1 and process_count_stored == process_count_jax // expansion_factor_real_data:
    # Scaling up to a larger number of hosts.(e.g., 32 files -> 64 processes)
    # In this case, a subset of hosts restore the data iterator.
    assert not isinstance(
        data_iterator, list
    ), "when expansion_factor_real_data > 1, the data iterator should not be a list."
    grain_restore_args = GrainCheckpointRestore(
        data_iterator.local_iterator, process_index=jax.process_index(), process_count=process_count_stored
    )

  else:
    # Case 4: Mismatch
    raise ValueError(
        f"Error restoring Grain checkpoint in {directory}: "
        f"The number of stored checkpoint files ({process_count_stored}) "
        f"is incompatible with the number of JAX processes ({process_count_jax}). "
        "If you are resuming training with a different number of chips, see instructions in "
        "https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/data_input_pipeline/"
        "data_input_grain.md#using-grain"
    )

  # Call restore once with the composed arguments
  restored_state = checkpoint_manager.restore(step, args=Composite(items=checkpoint_args, iter=grain_restore_args))
  return (restored_state, None)


def load_state_if_possible(
    checkpoint_manager: CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | list[MultiHostDataLoadIterator] | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    checkpoint_storage_concurrent_gb: int,
    abstract_unboxed_pre_state: train_state.TrainState,
    enable_single_replica_ckpt_restoring: bool | None = False,
    dataset_type: str | None = "tfds",
    step: int = -1,  # -1 means latest
    use_ocdbt=True,
    use_zarr3=True,
    enable_orbax_v1=False,
    checkpoint_conversion_fn=None,
    source_checkpoint_layout="orbax",
    expansion_factor_real_data: int = -1,
):
  """Loads TrainState as possible from the inputs.

  Args:
    checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
      that TrainState. This enables a full reload of a run in progress.
    load_parameters_from_path: if there is no checkpoint in the checkpoint
      manager, load parameters from a parameter only checkpoint at this path.
    load_full_state_from_path: if there is no checkpoint in the checkpoint
      manager, load full state from a full state checkpoint at this path.
    abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
      matches type against.
    enable_single_replica_ckpt_restoring: bool flag for restoring checkpoint
      with SingleReplicaArrayHandler
    checkpoint_storage_concurrent_gb: concurrent GB for checkpoint byte I/O.
    enable_orbax_v1: bool flag for enabling Orbax v1.
    checkpoint_conversion_fn: function for converting checkpoint to Orbax v1.
    source_checkpoint_layout: Optional checkpoint context to use for loading,
    provided in string format with the default being "orbax".

  Returns:
    A tuple of (train_state, train_state_params) where full_train_state captures
     a full reload and train_state_params just the params for a partial reload.
     At most one will be non-None. Both can be None if neither checkpoint is
     set.
  """

  if checkpoint_manager is not None:
    max_logging.log("checkpoint manager exists so trying to load this run's existing checkpoint")

    step = checkpoint_manager.latest_step() if step < 0 else step
    if step is not None:
      max_logging.log(f"restoring from this run's directory step {step}")

      # Check whether the mesh actually has multiple replicas along axis 0.
      # If all devices are in a single replica, SingleReplicaArrayHandler
      # raises InvalidShardingError — fall back to normal restore.
      _sr_effective = enable_single_replica_ckpt_restoring
      if _sr_effective:
        _first_leaf = jax.tree_util.tree_leaves(abstract_unboxed_pre_state)[0]
        if _first_leaf.sharding.mesh.devices.shape[0] <= 1:
          max_logging.log(
              "enable_single_replica_ckpt_restoring=True but mesh has only 1 "
              f"replica (shape[0]={_first_leaf.sharding.mesh.devices.shape[0]}). "
              "Falling back to normal all-replica restore.")
          _sr_effective = False

      def map_to_pspec(data):
        if not _sr_effective:
          return ocp.type_handlers.ArrayRestoreArgs(sharding=data.sharding)
        pspec = data.sharding.spec
        mesh = data.sharding.mesh
        replica_axis_index = 0
        replica_devices = _replica_devices(mesh.devices, replica_axis_index)
        replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
        single_replica_sharding = jax.sharding.NamedSharding(replica_mesh, pspec)

        return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
            sharding=jax.sharding.NamedSharding(mesh, pspec),
            single_replica_sharding=single_replica_sharding,
            global_shape=data.shape,
            dtype=data.dtype,
        )

      if _sr_effective:
        # Cache the original ArrayHandler so we can restore it after the
        # single-replica restore completes (see finally block below).
        original_array_handler = ocp.type_handlers.get_type_handler(jax.Array)
        single_replica_handler = ocp.type_handlers.SingleReplicaArrayHandler(
            replica_axis_index=0,
            broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
        )
        ocp.type_handlers.register_type_handler(
            jax.Array, single_replica_handler, override=True)

        # Monkeypatch orbax to use direct RCCL broadcast with explicit
        # communicator destroy + gc.collect() (falls back to Orbax default if
        # RCCL ctypes unavailable).  The default orbax broadcast creates RCCL
        # communicators via JAX/XLA that are cached with persistent proxy
        # threads, degrading training TGS.
        _install_custom_broadcast()

      restore_args = jax.tree_util.tree_map(map_to_pspec, abstract_unboxed_pre_state)
      checkpoint_args = ocp.args.PyTreeRestore(
          item=abstract_unboxed_pre_state, restore_args=restore_args)

      # try/finally guarantees that SingleReplicaArrayHandler and the
      # custom broadcast monkeypatch are always cleaned up, even if
      # restore() raises.  SingleReplicaArrayHandler is restore-only;
      # leaving it registered corrupts saves ("No ArrayMetadata found").
      try:
        match (checkpoint_manager, dataset_type, data_iterator):
          # Case 1: EmergencyCheckpointManager or EmergencyReplicatorCheckpointManager
          case (checkpoint_manager, _, _) if isinstance(
              checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)
          ):
            return (
                checkpoint_manager.restore(step, args=Composite(state=checkpoint_args)).state,
                None,
            )
          # Case 2: grain dataset with iterator checkpoint
          case (
              checkpoint_manager,
              dataset_type,
              data_iterator,
          ) if (
              dataset_type == "grain"
              and data_iterator
              and not isinstance(data_iterator, PlaceHolderDataIterator)
              and (checkpoint_manager.directory / str(step) / "iter").exists()
          ):
            return _restore_grain_iterator(
                checkpoint_manager, step, data_iterator, checkpoint_args, expansion_factor_real_data
            )
          # Case 3: Default/Fallback
          case _:
            return (checkpoint_manager.restore(step, args=Composite(items=checkpoint_args)), None)
      finally:
        if _sr_effective:
          ocp.type_handlers.register_type_handler(
              jax.Array, original_array_handler, override=True)
          _uninstall_custom_broadcast()
          gc.collect()

  if load_parameters_from_path != "":
    restored_params = load_params_from_path(
        load_parameters_from_path,
        abstract_unboxed_pre_state.params,
        checkpoint_storage_concurrent_gb,
        use_ocdbt=use_ocdbt,
        use_zarr3=use_zarr3,
    )
    return None, restored_params
  elif load_full_state_from_path != "":
    max_logging.log(f"Loading full state from path: {load_full_state_from_path}")
    restored_state = _load_full_state_from_path(
        path=load_full_state_from_path,
        abstract_unboxed_pre_state=abstract_unboxed_pre_state,
        enable_orbax_v1=enable_orbax_v1,
        checkpoint_conversion_fn=checkpoint_conversion_fn,
        source_checkpoint_layout=source_checkpoint_layout,
    )
    return {"items": restored_state}, None
  else:
    max_logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def setup_checkpoint_logger(config) -> Any | None:  # pytype: disable=attribute-error
  """Setup checkpoint logger.
  Args:
    config
  Returns:
    CloudLogger
  """
  orbax_cloud_logger = None
  max_logging.log("Setting up checkpoint logger...")
  if config.enable_checkpoint_cloud_logger:
    logger_name = f"goodput_{config.run_name}"
    orbax_cloud_logger = ocp.logging.CloudLogger(
        options=ocp.logging.CloudLoggerOptions(job_name=config.run_name, logger_name=logger_name)
    )
    max_logging.log("Successfully set up checkpoint cloud logger.")

  return orbax_cloud_logger


def load_params_from_path(
    load_parameters_from_path, abstract_unboxed_params, checkpoint_storage_concurrent_gb, use_ocdbt=True, use_zarr3=True
):
  """Load decode params from checkpoint at specified path."""
  assert load_parameters_from_path, "load_parameters_from_path is not defined."
  max_logging.log(f"restoring params from {load_parameters_from_path}")

  # *_concurrent_gb should be set for large models, the default is 96.
  max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")
  ckptr = ocp.Checkpointer(
      ocp.PyTreeCheckpointHandler(
          restore_concurrent_gb=checkpoint_storage_concurrent_gb,
          save_concurrent_gb=checkpoint_storage_concurrent_gb,
          use_ocdbt=use_ocdbt,
          use_zarr3=use_zarr3,
      )
  )

  # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
  # Rather than pass the entire abstract state, which could unnecessarily restore opt_state and such and waste
  # memory, we instead specify here that we are just restoring the params field of the checkpoint
  # (which itself may be a dictionary containing a key named 'params').
  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_unboxed_params)
  restored = ckptr.restore(
      epath.Path(load_parameters_from_path),
      item={"params": abstract_unboxed_params},
      transforms={},
      restore_args={"params": restore_args},
  )
  return restored["params"]


def save_params_to_path(checkpoint_dir, params, use_ocdbt=True, use_zarr3=True):
  """Save decode params in checkpoint at specified path."""
  assert checkpoint_dir, "checkpoint_dir is not defined."
  print(f"Saving quantized params checkpoint with use_ocdbt = {use_ocdbt} and use_zarr3 = {use_zarr3}")
  orbax_checkpointer = ocp.PyTreeCheckpointer(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)
  orbax_checkpointer.save(checkpoint_dir, {"params": params}, force=True)
  print(f"Quantized params checkpoint saved at: {checkpoint_dir}")


def maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None):
  """Save checkpoint if checkpointing is enabled."""
  if checkpoint_manager is None:
    return

  # Determine the effective step for saving a checkpoint.
  # If 'step' is not provided, this call is for a potential final checkpoint
  # and use the last completed step from the state.
  actual_step = (int(state.step) - 1) if step is None else int(step)

  # Determine if a checkpoint save should be forced, overriding the usual `config.checkpoint_period` logic.
  # This occurs if this function was called:
  # without an explicit 'step' (implying it's a checkpoint save for final step),
  # AND the 'actual_step' is a valid step,
  # AND it's not a step that would normally trigger a checkpoint save.
  force_ckpt_save = step is None and actual_step != -1 and (actual_step % config.checkpoint_period != 0)

  try:
    checkpoint_saved = save_checkpoint(checkpoint_manager, actual_step, state, config, data_iterator, force_ckpt_save)
    if checkpoint_saved:
      print_save_message(actual_step, config.async_checkpointing)
  except Exception as e:
    raise exceptions.StopTraining(f"Checkpointing failed. {str(e)}") from e

  # Wait for any pending checkpoint save to finish during preemption or final step save
  if force_ckpt_save or checkpoint_manager.reached_preemption(actual_step):
    checkpoint_manager.wait_until_finished()

  # Raise exception upon preemption
  if checkpoint_manager.reached_preemption(actual_step):
    raise exceptions.StopTraining("Job is preempted.")


def save_checkpoint(checkpoint_manager, step, state, config=None, data_iterator=None, force=False):
  """Wrapper for saving checkpoint."""
  if config and config.enable_checkpointing:
    if (
        force
        or (step % config.checkpoint_period == 0)
        or (config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0)
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      # We block here on the step finishing so that our checkpointing metrics
      # measure only checkpointing time, not training time.
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )

  # specify chunk_byte_size to force orbax to control maximum file size in checkpoint
  chunk_byte_size = (
      config.checkpoint_storage_target_data_file_size_bytes if config else DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  )

  checkpoint_args = ocp.args.PyTreeSave(
      item=state,
      save_args=jax.tree.map(lambda _: ocp.SaveArgs(chunk_byte_size=chunk_byte_size), state),
      ocdbt_target_data_file_size=chunk_byte_size,
  )
  save_args_composite = {"items": checkpoint_args}

  if config and config.dataset_type == "grain" and not isinstance(data_iterator, PlaceHolderDataIterator):
    if not isinstance(data_iterator, list):
      data_iterator = [data_iterator]
    grain_iters_to_save = []
    process_count_total = jax.process_count() * len(data_iterator)
    if config.expansion_factor_real_data > 1:
      process_count_total = process_count_total // config.expansion_factor_real_data
    for i, data_iter in enumerate(data_iterator):
      process_index = jax.process_index() + i * jax.process_count()
      grain_iters_to_save.append((data_iter.local_iterator, process_index, process_count_total))
    save_args_composite["iter"] = GrainCheckpointSave(item=grain_iters_to_save)

  match (checkpoint_manager, config, data_iterator):
    case (checkpoint_manager, _, _) if isinstance(
        checkpoint_manager, (EmergencyCheckpointManager, EmergencyReplicatorCheckpointManager)
    ):
      replicator_error_handler(config)
      return checkpoint_manager.save(step, args=Composite(state=checkpoint_args), force=force)
    case _:
      return checkpoint_manager.save(step, args=Composite(**save_args_composite), force=force)
