#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="$HOME/fp8-env"

# Create venv
uv venv "$ENV_DIR" --python 3.12
source "$ENV_DIR/bin/activate"

# PyTorch for aarch64 + CUDA 13.0 (PyPI gives CPU-only on aarch64)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# llmcompressor from main (latest TF5-compatible)
uv pip install git+https://github.com/vllm-project/llm-compressor.git@main

# Force TF5 + modern huggingface-hub back (llmcompressor pins old versions)
uv pip install git+https://github.com/huggingface/transformers.git@main --no-deps
uv pip install --upgrade huggingface-hub --no-deps

# Accelerate + upload tools
uv pip install accelerate hf_xet hf-transfer

# Verify
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}')
import transformers; print(f'transformers {transformers.__version__}')
import llmcompressor; print(f'llmcompressor {llmcompressor.__version__}')
print('All good.')
"
