export JAX_ENABLE_COMPILATION_CACHE=0
unset JAX_COMPILATION_CACHE_DIR
export XLA_FLAGS="--xla_dump_to=./log/llama3/with_quant --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_pass_re=spmd-partitioning --xla_gpu_enable_cublaslt=true --xla_gpu_enable_triton_gemm=false"

python3 -m MaxText.train MaxText/configs/models/gpu/llama3_8b.yml quantization="fp8"
