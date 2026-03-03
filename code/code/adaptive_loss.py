"""
Standalone adaptive loss function implementation
"""

import numpy as np
import torch

def adaptive_weights_numpy(f2_values: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Compute adaptive weights using NumPy"""
    f2_norm = np.linalg.norm(f2_values)
    numerator = f2_norm + 2 * tau
    denominator = 2 * (f2_norm + tau) ** 2
    return (1 + tau) * numerator / denominator

def adaptive_weights_torch(f2_values: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Compute adaptive weights using PyTorch"""
    f2_norm = torch.norm(f2_values, p=2)
    numerator = f2_norm + 2 * tau
    denominator = 2 * (f2_norm + tau) ** 2
    return (1 + tau) * numerator / denominator

def adaptive_loss(x: np.ndarray, tau: float = 1.0) -> float:
    """
    Compute adaptive loss value
    
    Args:
        x: Input array
        tau: Adaptive parameter
        
    Returns:
        Adaptive loss value
    """
    x_norm = np.linalg.norm(x)
    return (1 + tau) * (x_norm + 2 * tau) / (2 * (x_norm + tau))
