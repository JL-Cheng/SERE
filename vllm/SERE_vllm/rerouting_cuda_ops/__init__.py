"""
High-performance CUDA kernels for SERE in MoE models.
"""

import torch
from typing import Tuple

# Import the compiled CUDA extension directly
from . import rerouting_ops


def rerouting_ops_cuda(
    topk_weights: torch.Tensor, 
    topk_ids: torch.Tensor, 
    similarity_matrix: torch.Tensor, 
    select_top_k: int = 1,
    high_mask_cache: torch.Tensor = None,
    expert_mapping_cache: torch.Tensor = None,
    threshold: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-optimized Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models.
    
    Args:
        topk_weights: Top-k routing weights [num_tokens, top_k]
        topk_ids: Top-k expert indices [num_tokens, top_k] 
        similarity_matrix: Expert similarity matrix [num_experts, num_experts]
        select_top_k: Number of top primary experts to keep unchanged
        threshold: Similarity threshold - if best similarity is below this, route to original expert
        
    Returns:
        Tuple of (topk_weights, rerouted_topk_ids)
    """
    
    # Ensure inputs are on CUDA
    if not topk_weights.is_cuda:
        raise RuntimeError("topk_weights must be on CUDA device for CUDA operations")
    if not topk_ids.is_cuda:
        raise RuntimeError("topk_ids must be on CUDA device for CUDA operations")
    if not similarity_matrix.is_cuda:
        raise RuntimeError("similarity_matrix must be on CUDA device for CUDA operations")
    
    # Ensure correct data types
    topk_ids = topk_ids.to(torch.long)
    
    # Apply CUDA kernel for expert rerouting - pass pre-allocated tensors if available
    rerouted_topk_ids = rerouting_ops.reroute(
        topk_weights, topk_ids, similarity_matrix, select_top_k,
        high_mask_cache, expert_mapping_cache, threshold
    )
    
    return topk_weights, rerouted_topk_ids