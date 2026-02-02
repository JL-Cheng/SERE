# vLLM Plugin for SERE

This repository provides a CUDA‑based implementation of the SERE method, integrated into vLLM to enable efficient batch decoding for MoE models.

## Requirements

- Python 3.10.16
- PyTorch 2.6.0
- CUDA 12.4
- Transformers 4.52.3
- vLLM 0.8.4
- GCC/G++ 7.0+ (for compiling CUDA extensions, C++17 support required)

## Installation

Install the package using pip:

```bash
pip install .
```

**Important**: SERE requires vLLM v0 backend. Set the environment variable before running:

```bash
export VLLM_USE_V1=0
```

## Usage

**To enable SERE method, simply add the `hf_overrides` configuration as shown in the examples below.**

### Offline Inference

Use the following example for offline batch inference:

```python
import os
os.environ["VLLM_USE_V1"] = "0"  # Use vLLM v0 backend

from vllm import LLM, SamplingParams
from vllm.config import PoolerConfig

# Initialize vLLM offline inference engine for Qwen1.5-MoE
llm = LLM(
    model="path/to/your/model",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    hf_overrides={"architectures": ["Qwen2MoeForCausalLMSERE"], "select_top_k": 1, 'threshold': 0.1},
)

# Alternative configurations:

# For DeepSeek-V2-Lite-Chat:
# llm = LLM(
#     model="path/to/your/model",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.9,
#     trust_remote_code=True,
#     hf_overrides={"architectures": ["DeepseekV2ForCausalLMSERE"], "select_top_k": 1, 'threshold': 0.0},
# )

# For Qwen3-30B:
# llm = LLM(
#     model="path/to/your/model",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.9,
#     trust_remote_code=True,
#     hf_overrides={"architectures": ["Qwen3MoeForCausalLMSERE"], "select_top_k": 2, 'threshold': 0.0},
# )

# Format input text with prompt template
prompts = [
    "Hello, my name is",
    "The weather today is",
    "The future of technology is",
    "请写一个关于AI的故事。",
]

# Create sampling parameters
sampling_params = SamplingParams(
    max_tokens=50,
    temperature=0.0,
    prompt_logprobs=0,
)

# Perform batch inference
outputs = llm.generate(prompts, sampling_params)

# Process and display results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
```

### Online Deployment

For online deployment via API server, use the following command:

```bash
export VLLM_USE_V1=0  # Use vLLM v0 backend

vllm serve \
path/to/your/model \
--trust-remote-code \
--tensor-parallel-size 1 \
--disable-log-requests \
--max-model-len 4096 \
--gpu-memory-utilization 0.95 \
--hf-overrides '{"architectures": ["Qwen2MoeForCausalLMSERE"], "select_top_k": 2, "threshold": 0.0}'
```

## Configuration Parameters

- `select_top_k`: Number of primary experts to select for retaining
- `threshold`: Similarity threshold value for retaining critical experts
- `architectures`: Specifies the SERE architecture variant (e.g., "Qwen2MoeForCausalLMSERE", "DeepseekV2ForCausalLMSERE", "Qwen3MoeForCausalLMSERE")


Adjust these parameters based on your specific model and performance requirements. You can also refer to these architectures to adapt your own model accordingly.