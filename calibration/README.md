# Expert Similarity Calibration

This directory contains tools for computing and analyzing expert similarity matrices in Mixture of Experts (MoE) models. The system supports multiple model architectures and similarity computation methods.

## 📁 Directory Structure

```
calibration/
├── README.md                         # This file
├── cal_expert_similarity.py         # Main script for similarity calculation
├── utils.py                         # Similarity computation utilities
├── adapted_modeling_deepseek.py     # Modified DeepSeek-V2 model with similarity computation
├── adapted_modeling_qwen2_moe.py    # Modified Qwen2-MoE model with similarity computation
├── adapted_modeling_qwen3_moe.py    # Modified Qwen3-MoE model with similarity computation
└── configuration_deepseek.py        # DeepSeek-V2 configuration
```

## 🎯 Purpose

The calibration system allows you to:
- **Compute similarity matrices** between experts in MoE layers using various metrics
- **Analyze expert redundancy** and specialization patterns
- **Save enhanced models** with computed similarity matrices for further use
- **Generate statistics** about expert similarity distributions

## 🚀 Quick Start

### Basic Usage

```bash
python cal_expert_similarity.py \
    --model_type model_type \
    --model_path /path/to/original/model \
    --output_path /path/to/output/model \
    --data_path /path/to/calibration/data.parquet \
    --similarity_method similarity_method \
    --kernel kernel_for_cka \
    --batch_size batch_size \
    --max_len max_len
```

### Example Commands

#### For Qwen2-MoE Model
```bash
python cal_expert_similarity.py \
    --model_type qwen2_moe \
    --model_path Qwen/Qwen1.5-MoE-A2.7B \
    --output_path ./output/qwen2_moe_similarity \
    --data_path ./data/calibration_data.parquet \
    --similarity_method cka \
    --kernel linear \
    --batch_size 100 \
    --max_len 128
```

#### For DeepSeek-V2 Model
```bash
python cal_expert_similarity.py \
    --model_type deepseek_v2 \
    --model_path deepseek-ai/DeepSeek-V2-Lite \
    --output_path ./output/deepseek_v2_similarity \
    --data_path ./data/calibration_data.parquet \
    --similarity_method cosine \
    --batch_size 50 \
    --max_len 256
```

#### For Qwen3-MoE Model
```bash
python cal_expert_similarity.py \
    --model_type qwen3_moe \
    --model_path Qwen/Qwen3-MoE-15B-A2B \
    --output_path ./output/qwen3_moe_similarity \
    --data_path ./data/calibration_data.parquet \
    --similarity_method frobenius \
    --batch_size 32 \
    --max_len 512
```

## ⚙️ Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model_type` | Model architecture type | `qwen2_moe`, `qwen3_moe`, `deepseek_v2` |
| `--model_path` | Path to the original model | `Qwen/Qwen1.5-MoE-A2.7B` |
| `--output_path` | Output directory for enhanced model | `./output/model_with_similarity` |
| `--data_path` | Path to calibration data (parquet format) | `./data/calibration_data.parquet` |

### Optional Parameters

| Parameter | Default | Description | Options |
|-----------|---------|-------------|---------|
| `--similarity_method` | `frobenius` | Similarity computation method | `cka`, `cosine`, `frobenius` |
| `--kernel` | `linear` | Kernel type for CKA method | `linear`, `rbf`, `polynomial` |
| `--batch_size` | `200` | Number of texts to process | Any integer |
| `--max_len` | `64` | Maximum sequence length | Any integer |

## 📊 Similarity Methods

### 1. Centered Kernel Alignment (CKA)
- **Method**: `cka`
- **Kernels**: `linear`, `rbf`, `polynomial`
- **Description**: Measures linear and non-linear similarities between expert representations
- **Best for**: Understanding functional similarities between experts

```bash
--similarity_method cka --kernel linear
```

### 2. Cosine Similarity
- **Method**: `cosine`
- **Description**: Measures angular similarity between expert outputs
- **Best for**: Direction-based similarity analysis

```bash
--similarity_method cosine
```

### 3. Frobenius Norm
- **Method**: `frobenius`
- **Description**: Measures L2 distance-based similarity between expert outputs
- **Best for**: Magnitude-aware similarity analysis

```bash
--similarity_method frobenius
```

## 📋 Data Format

### Input Data
The calibration data should be in **Parquet format** with at least a `text` column:

```python
import pandas as pd

# Example data structure
data = pd.DataFrame({
    'text': [
        "This is sample text for calibration...",
        "Another example sentence...",
        # ... more texts
    ]
})
data.to_parquet('calibration_data.parquet')
```

### Text Filtering
The system automatically filters texts based on token length:
- Only texts with `token_length > max_len` are used
- This ensures sufficient context for meaningful expert activation

## 💾 Output Structure

After running the script, you'll get:

```
output_path/
├── config.json                      # Model configuration
├── model.safetensors                # Enhanced model weights
├── tokenizer.json                   # Tokenizer files
├── tokenizer_config.json
├── special_tokens_map.json
└── similarity_matrices.pt           # Computed similarity matrices
```

### Similarity Matrices Format
```python
import torch

# Load similarity matrices
similarity_matrices = torch.load('similarity_matrices.pt')

# Structure: {layer_index: similarity_matrix}
# Example: {12: tensor([[1.0, 0.8, ...], [0.8, 1.0, ...], ...]), ...}

for layer_idx, matrix in similarity_matrices.items():
    print(f"Layer {layer_idx}: {matrix.shape}")
    # Output: Layer 12: torch.Size([8, 8])  # 8x8 for 8 experts
```