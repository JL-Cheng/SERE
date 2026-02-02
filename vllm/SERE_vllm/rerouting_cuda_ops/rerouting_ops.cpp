#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

torch::Tensor reroute_cuda(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids, 
    torch::Tensor similarity_matrix,
    int64_t select_top_k,
    torch::Tensor high_mask_cache,
    torch::Tensor expert_mapping_cache,
    double threshold
);


torch::Tensor reroute(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor similarity_matrix,
    int64_t select_top_k,
    torch::Tensor high_mask_cache,
    torch::Tensor expert_mapping_cache,
    double threshold) {
    
    TORCH_CHECK(topk_weights.is_contiguous(), "topk_weights must be contiguous");
    TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(similarity_matrix.is_contiguous(), "similarity_matrix must be contiguous");
    TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be 2D");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D");
    TORCH_CHECK(similarity_matrix.dim() == 2, "similarity_matrix must be 2D");
    
    return reroute_cuda(topk_weights, topk_ids, similarity_matrix, select_top_k, 
                        high_mask_cache, expert_mapping_cache, threshold);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original function for backward compatibility
    m.def("reroute", &reroute, 
          "Similarity-based Expert Re-routing for Efficient Batch Decoding",
          py::arg("topk_weights"),
          py::arg("topk_ids"), 
          py::arg("similarity_matrix"),
          py::arg("select_top_k"), 
          py::arg("high_mask_cache") = torch::Tensor(), 
          py::arg("expert_mapping_cache") = torch::Tensor(),
          py::arg("threshold") = 0.0);
}