# Overview

This repository is a fork of [MaxText](https://github.com/AI-Hypercomputer/maxtext) to enable [JaxPP](https://github.com/ROCm/jaxpp/) for ROCm.

## Run models with JaxPP

### Install JaxPP
Follow this page to enable JaxPP for ROCm: https://github.com/ROCm/jaxpp/

### Install MaxText
`pip install -r requirements_rocm.txt` (you many need to install other dependencies in your container)

`pip install . --no-deps`

### Run models

```
# llama 4
CONFIG_FILE=./scripts/llama4_proxy_config.sh bash scripts/test_1gpu_config.sh

# mistral
MODEL_CONFIG='model_name=mistral-7b override_model_config=True base_num_decoder_layers=2' bash scripts/test_1gpu_config.sh
```

You will find the jaxpp is enabled:
<img width="1057" height="409" alt="image" src="https://github.com/user-attachments/assets/e7cb3812-2608-4267-9ad8-a08852c5f468" />
