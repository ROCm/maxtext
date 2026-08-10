#!/bin/bash
# Times end-to-end MaxText train steps for the MoE backends at a chosen
# expert-parallel degree.
#
# per_device_batch_size stays at 8 regardless of EP, so each device keeps the
# same 16384 grouped-GEMM rows the single-device benchmark used and the arms
# stay comparable to it. Global batch therefore scales with the device count.
#
# `size` picks the model dims: `mixtral` is Mixtral-8x7B's own width and
# `toy` is the shrunk stand-in, whose expert GEMMs are ~1/14 the work and so
# understate any GEMM-side change.
#
# Usage: bench_moe_ep.sh <ep> <steps> [toy|mixtral]
set -u
EP=${1:-4}
STEPS=${2:-30}
SIZE=${3:-mixtral}

case $SIZE in
  toy)     DIMS="base_emb_dim=1024 base_mlp_dim=4096 base_moe_mlp_dim=4096
             base_num_query_heads=8 base_num_kv_heads=8" ;;
  mixtral) DIMS="base_emb_dim=4096 base_mlp_dim=14336 base_moe_mlp_dim=14336
             base_num_query_heads=32 base_num_kv_heads=8" ;;
  *) echo "unknown size '$SIZE' (want toy|mixtral)" >&2; exit 2 ;;
esac

# XLA's command buffer path recurses through the HIP runtime deep enough to
# blow the stack once the graph gets large, taking the process with it
# (SIGSEGV inside RocmCommandBuffer::LaunchGraph). It is also what made decode
# 1.5-1.9x slower in the inference campaign. Off for every arm, so the arms
# stay comparable and the larger sizes run at all.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

COMMON="src/maxtext/configs/base.yml dataset_type=synthetic
  steps=$STEPS per_device_batch_size=8 max_target_length=1024
  $DIMS head_dim=128 base_num_decoder_layers=2
  num_experts=8 num_experts_per_tok=2 sparse_matmul=true
  attention=dot_product decoder_block=mixtral
  enable_checkpointing=false scan_layers=false
  ici_expert_parallelism=$EP ici_fsdp_parallelism=1
  base_output_directory=/tmp/mt_bench"

# FP8_OPS scopes the qwix fp8 rule to a chosen op list; see bench_launcher.py.
# Unset means stock behaviour, which for `quantization=fp8_full` is every op.
run() {
  local tag=$1; local fp8_ops=$2; shift 2
  local name="${SIZE}_ep${EP}_$tag"
  FP8_OPS="$fp8_ops" PYTHONPATH=/jax_dir/jax-flydsl:/jax_dir/maxtext-mxfp8-train/src \
    timeout 2400 python3 bench_launcher.py $COMMON run_name="$name" megablox=false "$@" \
    > "/tmp/$name.log" 2>&1
  local rc=$?
  grep -oE "completed step: [0-9]+, seconds: [0-9.]+.*loss: [0-9.]+, lm" "/tmp/$name.log" \
    | sed "s/^/$tag /"
  echo "### $tag exit=$rc" >&2
}

FP8="quantization=fp8_full use_qwix_quantization=true"

# The arms differ only in how the three MoE expert GEMMs are contracted. bf16 is
# the reference; fp8_moe is the like-for-like comparison, since MXFP8 also
# touches nothing but those GEMMs. fp8_full quantizes attention and the dense
# projections too, so it is faster for reasons that have nothing to do with the
# MoE backend -- kept only to show that gap.
run bf16      ""                 # everything bf16
run fp8_moe   "gmm,ragged_dot"  $FP8
run fp8_full  ""                $FP8
run mxfp8     ""                 use_flydsl_moe=true
