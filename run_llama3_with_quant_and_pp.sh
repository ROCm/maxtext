export JAX_ENABLE_COMPILATION_CACHE=0
unset JAX_COMPILATION_CACHE_DIR

mkdir -p ./log/llama3/with_quant_and_pp

export JAX_ENABLE_COMPILATION_CACHE=0
unset JAX_COMPILATION_CACHE_DIR
export XLA_FLAGS="--xla_dump_to=./log/llama3/with_quant_and_pp --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_pass_re='spmd-partitioning|while-loop-all-reduce-code-motion' --xla_gpu_enable_cublaslt=true --xla_gpu_enable_triton_gemm=false"
TF_CPP_MIN_LOG_LEVEL=0
TF_CPP_MAX_VLOG_LEVEL=5
export TF_CPP_VMODULE=while_loop_all_reduce_code_motion=5,spmd_partitioner=5

python3 -m MaxText.train MaxText/configs/models/gpu/llama3_8b_pp.yml quantization="fp8" > ./log/llama3/with_quant_and_pp/run.log 2>&1
