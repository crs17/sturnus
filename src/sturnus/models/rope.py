from typing import Tuple

import matplotlib.pylab as plt
import torch


def calculate_thetas(embedding_dimension: int, context_size: int) -> Tuple[torch.Tensor]:
    """ Calculate rotation angles for each 2D slice along the per-head embedding dimension. """
    
    assert embedding_dimension % 2 == 0, 'Embedding dimension per head must be even'

    thetas = 10_000 ** (-2 * (torch.arange(1, embedding_dimension / 2 + 1).float() - 1 ) / embedding_dimension)  # [P/2]

    positions = torch.arange(context_size, device=thetas.device).float()  # [C]
    thetas_expanded = torch.outer(positions, thetas)

    c_values = torch.polar(torch.ones_like(thetas_expanded), thetas_expanded)  # [C, P/2] 

    return c_values, thetas


def apply_rope(x, c_values):
    # The input x is expected to have the shape [B, C, H, P]
    # Break up the embedding dimension into pairs or 2D vectors
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, C, H, P/2, 2]

    # Convert the 2D vectors into complex numbers
    x_c = torch.view_as_complex(x_pairs)  # [B, C, H, P/2]

    # Ensure that the shape of the precalculated rotation angles matches the input
    assert c_values.shape == (x_c.shape[1], x_c.shape[-1])

    # Make the c_values broadcastable of x:
    new_shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x_c.shape)]  # [1, C, 1, P/2]
    c_values_reshaped = c_values.view(*new_shape)  # [1, C, 1, P/2]

    # Apply the rotation
    x_c_rot = x_c * c_values_reshaped  # [B, C, H, P/2]

    # Convert back to real numbers and flatten the last dimension to collect all pairs together:
    x_rot = torch.view_as_real(x_c_rot).flatten(start_dim=3)  # [B, C, H, P]

    return x_rot
