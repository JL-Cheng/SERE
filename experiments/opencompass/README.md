# OpenCompass Evaluation Experiments

This directory contains OpenCompass evaluation configurations for different MoE models.

## Overview

The project evaluates three different SERE-accelerated models using the OpenCompass framework:
- **DeepSeek V2** with SERE modifications
- **Qwen 1.5 MoE** with SERE modifications
- **Qwen 3 MoE** with SERE modifications

## Files Structure

```
├── eval_deepseek_v2.py    # Configuration for DeepSeek V2 SERE evaluation
├── eval_qwen1_5.py        # Configuration for Qwen 1.5 MoE SERE evaluation
├── eval_qwen3.py          # Configuration for Qwen 3 MoE SERE evaluation
├── run_exps.sh            # Batch script to run all evaluations
└── README.md              # This file
```

## Evaluation Datasets

All configurations evaluate models on the same comprehensive benchmark suite:

### Exam Category
- **CMMLU**: Chinese Massive Multi-task Language Understanding
- **BoolQ**: Boolean Questions from SuperGLUE
- **BBH**: Big-Bench Hard reasoning tasks

### Math Category
- **MATH**: Competition-level mathematics problems
- **GSM8K**: Grade school math word problems
- **MATH401**: Extended mathematics dataset

### Code Category
- **HumanEval**: Python code generation benchmark
- **MBPP**: Mostly Basic Python Programming problems

## Model Configurations

### SERE Architecture Modifications

Each model uses custom SERE (Sparse Expert Routing Enhancement) architectures:
- DeepSeek V2: `DeepseekV2ForCausalLMSERE`
- Qwen 1.5: `Qwen2MoeForCausalLMSERE`
- Qwen 3: `Qwen3MoeForCausalLMSERE`

### SERE Parameters
All models are configured with:
- `select_top_k` - Number of top experts to select
- `threshold` - Similarity threshold for expert re-routing



## Usage

### Prerequisites
1. Install OpenCompass framework
2. Ensure model weights are available at the specified paths
3. Configure CUDA/GPU environment for VLLM

### Running Evaluations

#### Individual Model Evaluation
```bash
# DeepSeek V2 SERE
opencompass eval_deepseek_v2.py --work-dir ./results/deepseek_v2/ --mode all --reuse

# Qwen 1.5 MoE SERE
opencompass eval_qwen1_5.py --work-dir ./results/qwen1_5/ --mode all --reuse

# Qwen 3 MoE SERE
opencompass eval_qwen3.py --work-dir ./results/qwen3/ --mode all --reuse
```

#### Batch Evaluation
```bash
chmod +x run_exps.sh
./run_exps.sh
```

### Configuration Details

#### Hardware Requirements
- **GPU Memory**: 90% utilization configured
- **Tensor Parallel**: Single GPU deployment
- **Batch Size**: 16 for all models
- **Workers**: 8 inference workers, up to 128 evaluation workers

#### Output Configuration
- **DeepSeek V2 & Qwen 1.5**: Max 1024 tokens
- **Qwen 3**: Max 2048 tokens



## Evaluation Metrics

The summarizer provides hierarchical reporting:
- **Overall**: Aggregate across all categories
- **Category Scores**: Exam, Math, Code performance
- **Individual Dataset Scores**: Detailed breakdown per benchmark
