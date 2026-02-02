#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>


template<typename scalar_t>
__global__ void reroute_kernel(
    const int64_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ similarity_matrix,
    const bool* __restrict__ high_mask,
    int64_t* __restrict__ output_ids,
    const int num_tokens,
    const int top_k,
    const int select_top_k,
    const int num_experts,
    const scalar_t threshold) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int reroute_slots = top_k - select_top_k;
    if (reroute_slots <= 0) return;
    
    int token_idx = idx / reroute_slots;
    int slot_in_reroute = idx % reroute_slots;
    int expert_slot = select_top_k + slot_in_reroute;
    
    if (token_idx >= num_tokens || expert_slot >= top_k) return;
    
    int input_idx = token_idx * top_k + expert_slot;
    int64_t original_expert = topk_ids[input_idx];
    
    if (original_expert < 0 || original_expert >= num_experts) {
        output_ids[input_idx] = 0;
        return;
    }
    
    // Early return optimization: if original expert is already in high_mask, route to itself.
    if (high_mask[original_expert]) {
        output_ids[input_idx] = original_expert;
        return;
    }
    
    scalar_t best_similarity = static_cast<scalar_t>(-1e9);
    int best_expert = 0;
    
    for (int high_expert = 0; high_expert < num_experts; high_expert++) {
        if (high_mask[high_expert]) {
            scalar_t similarity = similarity_matrix[original_expert * num_experts + high_expert];
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_expert = high_expert;
            }
        }
    }
    
    // Apply threshold: if best similarity is below threshold, route to original expert
    if (threshold > 0.0 && best_similarity < threshold) {
        output_ids[input_idx] = original_expert;
    } else {
        output_ids[input_idx] = best_expert;
    }
}

torch::Tensor reroute_cuda(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor similarity_matrix,
    int64_t select_top_k,
    torch::Tensor high_mask_cache,
    torch::Tensor expert_mapping_cache,
    double threshold) {
    
    const int num_tokens = topk_weights.size(0);
    const int top_k = topk_weights.size(1);
    const int num_experts = similarity_matrix.size(0);
    
    if (select_top_k <= 0 || select_top_k >= top_k) {
        return topk_ids;
    }
    
    // Use merged kernel for re-route
    {
        const int reroute_elements = num_tokens * (top_k - select_top_k);
        
        if (reroute_elements > 0) {
            // Create high_mask for merged kernel
            torch::Tensor high_mask;
            if (high_mask_cache.defined()) {
                high_mask = high_mask_cache;
                high_mask.zero_();
            } else {
                high_mask = torch::zeros({num_experts}, torch::dtype(torch::kBool).device(topk_weights.device()));
            }
            
            auto high_experts = topk_ids.slice(1, 0, select_top_k);
            high_mask.scatter_(0, high_experts.flatten(), true);
            
            const int threads_per_block = 256;
            const int reroute_blocks = (reroute_elements + threads_per_block - 1) / threads_per_block;
            
            // vLLM pattern: use CUDA guard and current stream for graph compatibility
            const at::cuda::OptionalCUDAGuard device_guard(topk_ids.device());
            const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            
            AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, similarity_matrix.scalar_type(), "reroute_kernel", [&] {
                reroute_kernel<scalar_t><<<reroute_blocks, threads_per_block, 0, stream>>>(
                    topk_ids.data_ptr<int64_t>(),
                    similarity_matrix.data_ptr<scalar_t>(),
                    high_mask.data_ptr<bool>(),
                    topk_ids.data_ptr<int64_t>(),  // Write back to original tensor
                    num_tokens,
                    top_k,
                    select_top_k,
                    num_experts,
                    static_cast<scalar_t>(threshold));
            });
        }
        
        return topk_ids;
    }
}

