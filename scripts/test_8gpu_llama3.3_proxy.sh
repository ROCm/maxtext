export JAX_USE_SHARDY_PARTITIONER=0
export JAXPP_ENABLE_LICM=1
export NVTE_FUSED_ATTN=1
# --xla_dump_hlo_pass_re=.*
export XLA_FLAGS="--xla_dump_hlo_as_html --xla_dump_hlo_as_text --xla_dump_to='./llama3-hlos-pp2' --xla_gpu_enable_latency_hiding_scheduler=true"
source scripts/llama3.3_proxy_config.sh

export PARALLELISM_CONFIG="
    dcn_pipeline_parallelism=1 ici_pipeline_parallelism=2
    ici_data_parallelism=1
    ici_context_parallelism=2
    ici_tensor_parallelism=2
    ici_fsdp_parallelism=1
    per_device_batch_size=1
    max_target_length=8192
"

export JAXPP_CONFIG="
    scan_layers=False
    jaxpp_remote=False
    use_jaxpp=True
    schedule=interleaved_1f1b
    num_pipeline_microbatches=4
    num_pipeline_repeats=1
    profiler=xplane
"

export N_PROCS=8
export N_GPUS=1

bash ./scripts/run_local_mc.sh
