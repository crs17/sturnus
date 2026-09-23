from typing import Tuple

import matplotlib.pylab as plt
import torch


def calculate_thetas(embedding_dimension: int, context_size: int, theta_zero: int = 10_000) -> Tuple[torch.Tensor]:
    """ Calculate rotation angles for each 2D slice along the per-head embedding dimension. """
    
    assert embedding_dimension % 2 == 0, 'Embedding dimension per head must be even'

    thetas = theta_zero ** (-2 * (torch.arange(1, embedding_dimension / 2 + 1).float() - 1 ) / embedding_dimension)  # [P/2]

    positions = torch.arange(context_size, device=thetas.device).float()  # [C]
    thetas_expanded = torch.outer(positions, thetas)

    c_values = torch.polar(torch.ones_like(thetas_expanded), thetas_expanded)  # [C, P/2] 

    return c_values, thetas


def calculate_rotations(x_c, x, c_values):
    # Ensure that the shape of the precalculated rotation angles matches the input
    assert c_values.shape == (x_c.shape[1], x_c.shape[-1]), f'c_values shape({c_values.shape}) do not match x_c shape ({x_c.shape})'

    # Make the c_values broadcastable of x:
    new_shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x_c.shape)]  # [1, C, 1, P/2]
    c_values_reshaped = c_values.view(*new_shape)  # [1, C, 1, P/2]

    # Apply the rotation
    x_c_rot = x_c * c_values_reshaped  # [B, C, H, P/2]

    return x_c_rot


def apply_rope(x, c_values):
    """ The original RoPE convention which pairs (x0, x1), (x2, x3) etc. """
    # The input x is expected to have the shape [B, C, H, P]
    # Break up the embedding dimension into pairs or 2D vectors.
    # # In the original RoPE convention, the pairs are (x0, x1), (x2, x3) etc.
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, C, H, P/2, 2]

    # Convert the 2D vectors into complex numbers
    x_c = torch.view_as_complex(x_pairs)  # [B, C, H, P/2]

    x_c_rot = calculate_rotations(x_c, x, c_values)

    # Convert back to real numbers and flatten the last dimension to collect all pairs together:
    x_rot = torch.view_as_real(x_c_rot).flatten(start_dim=3)  # [B, C, H, P]

    return x_rot


def apply_rope_half(x, c_values):
    """ RoPE convention of Llama-style models which pairs x[i] with x[i + P/2]"""
    # The input x is expected to have the shape [B, C, H, P]
    # Break up the embedding dimension into pairs or 2D vectors
    # In the Llama-style convention, the pairs are (x0, x[P/2]), (x1, x[P/2+1]) etc.
    x1, x2 = x.float().chunk(2, dim=-1)          # each [B, C, H, P/2]
    x_c = torch.complex(x1, x2)

    x_c_rot = calculate_rotations(x_c, x, c_values)

    # Convert back to real numbers and flatten the last dimension to collect all pairs together:
    x_rot = torch.cat([x_c_rot.real, x_c_rot.imag], dim=-1)

    return x_rot
