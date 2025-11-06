"""
Neural network weight initialization utilities
"""
import torch
import torch.nn as nn


def linear_weights_init(m):
    """Initialize linear layer weights"""
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -3e-3, 3e-3)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -3e-3, 3e-3)
