#!/usr/bin/env python3
"""
Similarity matrix computation utility functions
Includes CKA, cosine similarity, Frobenius similarity calculations
"""

import torch
import torch.nn.functional as F


def cka_kernel(X, Y, kernel='linear'):
    """Compute CKA kernel alignment

    Args:
        X: First representation matrix [n_samples, n_features]
        Y: Second representation matrix [n_samples, n_features]
        kernel: Kernel function type ('linear', 'rbf', 'polynomial')

    Returns:
        Kernel matrix [n_samples, n_samples]
    """
    if kernel == 'linear':
        return X @ Y.T
    elif kernel == 'rbf':
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        Y_norm = (Y ** 2).sum(dim=1, keepdim=True)
        distances = X_norm + Y_norm.T - 2 * X @ Y.T
        variance = torch.var(distances).to(distances.dtype)
        if variance > 1e-8:
            return torch.exp(-distances / (2 * variance))
        else:
            return torch.ones_like(distances)
    elif kernel == 'polynomial':
        return (X @ Y.T + 1) ** 2
    else:
        raise ValueError(f"Unsupported kernel function: {kernel}")


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    # Kc = K - row_mean - col_mean + total_mean
    mean_row = K.mean(dim=1, keepdim=True)
    mean_col = K.mean(dim=0, keepdim=True)
    mean_all = K.mean()
    return K - mean_row - mean_col + mean_all


def _cka_from_grams(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    # Center then normalized Frobenius inner-product (O(n^2), no n^3 ops)
    Kc = _center_gram((K + K.T) * 0.5)
    Lc = _center_gram((L + L.T) * 0.5)
    num = (Kc * Lc).sum()
    den = torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum()).clamp_min(1e-12)
    return torch.clamp(num / den, 0.0, 1.0)


def _cka_linear_features(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # Feature-space linear CKA: avoid Gram matrices entirely
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    Cxy = Yc.T @ Xc  # [dy, dx]
    Cxx = Xc.T @ Xc
    Cyy = Yc.T @ Yc
    num = (Cxy * Cxy).sum()
    den = torch.sqrt((Cxx * Cxx).sum() * (Cyy * Cyy).sum()).clamp_min(1e-12)
    return torch.clamp(num / den, 0.0, 1.0)


def _rff_features(X: torch.Tensor, num_features: int = 1024, gamma: float | None = None) -> torch.Tensor:
    # Random Fourier Features for RBF kernel approximation:
    # k(x,y) ≈ φ(x)^T φ(y), φ(x) = sqrt(2/m) cos(x W + b), W~N(0, 2γ I)
    d = X.shape[1]
    if gamma is None:
        # Light heuristic bandwidth; override if you know better
        gamma = 1.0 / max(d, 1)
    W = torch.randn(d, num_features, device=X.device, dtype=X.dtype) * (2.0 * gamma) ** 0.5
    b = 2.0 * torch.pi * torch.rand(1, num_features, device=X.device, dtype=X.dtype)
    Z = X @ W + b  # [n, m]
    return (2.0 / num_features) ** 0.5 * torch.cos(Z)


def centered_kernel_alignment(X, Y, kernel='linear'):
    """Compute CKA using accelerated paths
    - linear: Feature-space linear CKA (no Gram, fast)
    - rbf_rff: RFF approximation + linear CKA (approximate, fast)
    - rbf/polynomial: Computation via centered Gram (exact, O(n^2))

    Args:
        X: [n_samples, n_features]
        Y: [n_samples, n_features]
        kernel: ('linear', 'rbf', 'polynomial', 'rbf_rff')

    Returns:
        CKA similarity score (scalar tensor)
    """
    if kernel == 'linear':
        return _cka_linear_features(X, Y)

    if kernel in ('rbf_rff', 'rbf-rff'):
        # Map to random Fourier features then reuse fast linear CKA
        Zx = _rff_features(X)
        Zy = _rff_features(Y)
        return _cka_linear_features(Zx, Zy)

    # Exact kernel: construct Gram and use centered inner product to avoid O(n^3)
    K = cka_kernel(X, X, kernel)
    L = cka_kernel(Y, Y, kernel)
    return _cka_from_grams(K, L)


def cosine_similarity_matrix_gpu(expert_outputs, device='cuda'):
    """Compute cosine similarity matrix - GPU optimized version

    Args:
        expert_outputs: List of expert outputs, each element is [batch_size, seq_len, hidden_dim]
        device: Computing device

    Returns:
        Similarity matrix [n_experts, n_experts]
    """
    n_experts = len(expert_outputs)
    similarity_matrix = torch.zeros(n_experts, n_experts, device=device)
    
    # Move all expert outputs to specified device
    expert_outputs_gpu = [output.to(device) for output in expert_outputs]
    
    for i in range(n_experts):
        for j in range(i, n_experts):
            # Flatten to [batch_size * seq_len, hidden_dim]
            X = expert_outputs_gpu[i].view(-1, expert_outputs_gpu[i].shape[-1])
            Y = expert_outputs_gpu[j].view(-1, expert_outputs_gpu[j].shape[-1])
            
            # Compute cosine similarity
            sample_similarities = F.cosine_similarity(X, Y, dim=1)
            avg_sim = sample_similarities.mean()
            
            # Normalize to [0,1] range
            sim_normalized = (avg_sim + 1) / 2
            similarity_matrix[i, j] = similarity_matrix[j, i] = sim_normalized

    # Set diagonal to 1.0 (self-similarity set to 1.0)
    for i in range(n_experts):
        similarity_matrix[i, i] = 1.0
        
    return similarity_matrix


def frobenius_similarity_matrix_gpu(expert_outputs, device='cuda'):
    """Compute Frobenius norm-based similarity matrix - GPU optimized version

    Args:
        expert_outputs: List of expert outputs, each element is [batch_size, seq_len, hidden_dim]
        device: Computing device

    Returns:
        Similarity matrix [n_experts, n_experts]
    """
    n_experts = len(expert_outputs)
    similarity_matrix = torch.zeros(n_experts, n_experts, device=device)
    
    # Move all expert outputs to specified device
    expert_outputs_gpu = [output.to(device) for output in expert_outputs]
    
    # Compute Frobenius distances for all pairs
    distances = torch.zeros(n_experts, n_experts, device=device)
    for i in range(n_experts):
        for j in range(i, n_experts):
            diff = expert_outputs_gpu[i] - expert_outputs_gpu[j]
            frob_norm = torch.norm(diff, p='fro')
            distances[i, j] = distances[j, i] = frob_norm
    
    # Convert distances to similarity (smaller distance means higher similarity)
    max_distance = distances.max()
    
    for i in range(n_experts):
        for j in range(i, n_experts):
            if max_distance > 0:
                sim = 1 - (distances[i, j] / max_distance)
            else:
                sim = 1.0
            similarity_matrix[i, j] = similarity_matrix[j, i] = sim
    
    # Set diagonal to 1.0 (self-similarity set to 1.0)
    for i in range(n_experts):
        similarity_matrix[i, i] = 1.0
        
    return similarity_matrix


def cka_similarity_matrix_gpu(expert_outputs, kernel='linear', device='cuda'):
    """Compute CKA similarity matrix - GPU optimized version

    Args:
        expert_outputs: List of expert outputs, each element is [batch_size, seq_len, hidden_dim]
        kernel: CKA kernel function type ('linear', 'rbf', 'polynomial', 'rbf_rff')
        device: Computing device

    Returns:
        Similarity matrix [n_experts, n_experts]
    """
    n_experts = len(expert_outputs)
    similarity_matrix = torch.zeros(n_experts, n_experts, device=device)
    
    # Move all expert outputs to specified device
    expert_outputs_gpu = [output.to(device) for output in expert_outputs]
    
    for i in range(n_experts):
        for j in range(i, n_experts):
            # Flatten to [batch_size * seq_len, hidden_dim]
            X = expert_outputs_gpu[i].view(-1, expert_outputs_gpu[i].shape[-1])
            Y = expert_outputs_gpu[j].view(-1, expert_outputs_gpu[j].shape[-1])
            # Compute CKA similarity (centered_kernel_alignment now automatically selects fastest path based on kernel)
            cka_score = centered_kernel_alignment(X, Y, kernel)
            similarity_matrix[i, j] = similarity_matrix[j, i] = cka_score
    
    # Set diagonal to 1.0 (self-similarity set to 1.0)
    for i in range(n_experts):
        similarity_matrix[i, i] = 1.0
    
    return similarity_matrix


def compute_similarity_matrix_gpu(expert_outputs, method='cka', kernel='linear', device='cuda'):
    """Compute similarity matrix - GPU optimized version

    Args:
        expert_outputs: List of expert outputs, each element is [batch_size, seq_len, hidden_dim]
        method: Similarity computation method ('cka', 'cosine', 'frobenius')
        kernel: CKA kernel function type ('linear', 'rbf', 'polynomial', 'rbf_rff')
        device: Computing device

    Returns:
        Similarity matrix [n_experts, n_experts]

    Raises:
        ValueError: When method is not in the supported methods list
    """
    if method == 'cka':
        return cka_similarity_matrix_gpu(expert_outputs, kernel, device)
    elif method == 'cosine':
        return cosine_similarity_matrix_gpu(expert_outputs, device)
    elif method == 'frobenius':
        return frobenius_similarity_matrix_gpu(expert_outputs, device)
    else:
        raise ValueError(f"Unsupported similarity computation method: {method}. Supported methods: ['cka', 'cosine', 'frobenius']")


def get_similarity_statistics(similarity_matrix):
    """Get statistical information of similarity matrix

    Args:
        similarity_matrix: Similarity matrix [n_experts, n_experts]

    Returns:
        dict: Dictionary of statistical information
    """
    if similarity_matrix is None:
        return {"error": "Similarity matrix is None"}
    
    # Only consider upper triangular part (excluding diagonal)
    n = similarity_matrix.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    values = similarity_matrix[mask]
    
    stats = {
        "shape": similarity_matrix.shape,
        "mean": values.mean().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "median": values.median().item(),
        "n_pairs": len(values),
    }
    
    return stats


def print_similarity_statistics(similarity_matrices, layer_indices=None):
    """Print statistical information of similarity matrices

    Args:
        similarity_matrices: Dictionary of similarity matrices {layer_idx: matrix}
        layer_indices: List of layer indices (optional)
    """
    if not similarity_matrices:
        print("No similarity matrix data")
        return
    
    print("\nSimilarity Matrix Statistics:")
    print("-" * 60)
    
    for layer_idx, matrix in similarity_matrices.items():
        stats = get_similarity_statistics(matrix)
        
        if "error" in stats:
            print(f"Layer {layer_idx}: {stats['error']}")
            continue
        
        print(f"Layer {layer_idx}: shape {stats['shape']}")
        print(f"    Average similarity: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    Expert pairs: {stats['n_pairs']}")
        print()