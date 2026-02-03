export JAX_ENABLE_COMPILATION_CACHE=0
unset JAX_COMPILATION_CACHE_DIR

EXP_NAME="with_quant"

mkdir -p ./log/llama3/${EXP_NAME}
mkdir -p ./log/llama3/${EXP_NAME}/run_logs
mkdir -p ./log/llama3/${EXP_NAME}/rocprof

export XLA_FLAGS="--xla_dump_to=./log/llama3/${EXP_NAME}/hlo_dumps --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_gpu_enable_cublaslt=true --xla_gpu_enable_triton_gemm=false"
# TF_CPP_MIN_LOG_LEVEL=0
# TF_CPP_MAX_VLOG_LEVEL=5
# export TF_CPP_VMODULE=while_loop_all_reduce_code_motion=5,spmd_partitioner=5

# Use rocprof to trace GPU activities
# Options:
#   --hip-trace: trace HIP API and GPU kernel execution
#   --sys-trace: trace both HIP and HSA levels (more detailed)
#   -d: output directory for trace files
#   --stats: generate statistics CSV files
rocprofv3 -d ./log/llama3/${EXP_NAME}/rocprof --hip-trace --kernel-trace --rccl-trace --stats --output-format=pftrace \
    -- python3 -m MaxText.train MaxText/configs/models/gpu/llama3_8b.yml quantization="fp8" > ./log/llama3/${EXP_NAME}/run_logs/run.log 2>&1

# python3 -m MaxText.train MaxText/configs/models/gpu/llama3_8b.yml quantization="fp8" > ./log/llama3/${EXP_NAME}/run_logs/run.log 2>&1
