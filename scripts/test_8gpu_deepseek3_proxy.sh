source scripts/deepseek3_proxy_config.sh

export PARALLELISM_CONFIG="
    dcn_pipeline_parallelism=1 ici_pipeline_parallelism=2
    ici_data_parallelism=1
    ici_tensor_parallelism=2
    ici_expert_parallelism=2
    ici_fsdp_parallelism=1
"

export JAXPP_CONFIG="
    scan_layers=False
    use_jaxpp=True
    schedule=interleaved_1f1b
    num_pipeline_microbatches=4
    num_pipeline_repeats=1
"

export N_PROCS=2
export N_GPUS=4

bash ./scripts/run_local_mc.sh
