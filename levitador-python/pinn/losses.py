from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from model import TimeMLP


def _masked_mse(a: torch.Tensor,
                 b: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 reduction: str = "mean") -> torch.Tensor:
    """
    Mean squared error with optional boolean mask and reduction.
    a, b: tensors of same shape.
    mask: boolean tensor broadcastable to a/b; selects valid entries.
    reduction: 'mean' | 'sum' | 'none'.
    """
    if mask is None:
        mask = torch.isfinite(a) & torch.isfinite(b)
    if mask.ndim < a.ndim:
        # Broadcast mask to the last dimension if needed
        for _ in range(a.ndim - mask.ndim):
            mask = mask.unsqueeze(-1)
    valid = mask & torch.isfinite(a) & torch.isfinite(b)
    if not torch.any(valid):
        # No valid samples: return zero (keeps graph intact)
        return (a - b).detach().new_tensor(0.0, requires_grad=True)
    diff = (a - b)[valid]
    mse = (diff * diff)
    if reduction == "mean":
        return mse.mean()
    if reduction == "sum":
        return mse.sum()
    return mse  # 'none'


def loss_data(
    y_hat: torch.Tensor,
    i_hat: torch.Tensor,
    y_real: torch.Tensor,
    i_real: torch.Tensor,
    *,
    w_y: float = 1.0,
    w_i: float = 1.0,
    sigma_y: Optional[torch.Tensor] = None,
    sigma_i: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Data loss anchoring NN predictions to measurements.

    Loss_data = w_y * MSE((y_hat - y_real)/sigma_y) + w_i * MSE((i_hat - i_real)/sigma_i)

    - Inputs are expected in physical units (meters, amperes).
    - If sigma_y / sigma_i are provided, residuals are standardized.
    - mask can exclude invalid samples (e.g., saturated or corrupted points).
    - reduction in {'mean','sum','none'}.
    """
    # Standardize if sigmas are provided
    if sigma_y is not None:
        # Ensure broadcasting
        y_res = (y_hat - y_real) / sigma_y
    else:
        y_res = y_hat - y_real

    if sigma_i is not None:
        i_res = (i_hat - i_real) / sigma_i
    else:
        i_res = i_hat - i_real

    ly = _masked_mse(y_res, torch.zeros_like(y_res), mask=mask, reduction=reduction)
    li = _masked_mse(i_res, torch.zeros_like(i_res), mask=mask, reduction=reduction)
    return w_y * ly + w_i * li


def loss_data_from_model(
    model: nn.Module,
    t_norm: torch.Tensor,
    y_real: torch.Tensor,
    i_real: torch.Tensor,
    *,
    w_y: float = 1.0,
    w_i: float = 1.0,
    sigma_y: Optional[torch.Tensor] = None,
    sigma_i: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Convenience wrapper that queries the model and calls loss_data.
    The model must return (y_hat, i_hat) in physical units.
    """
    y_hat, i_hat = model(t_norm)
    return loss_data(
        y_hat, i_hat, y_real, i_real,
        w_y=w_y, w_i=w_i,
        sigma_y=sigma_y, sigma_i=sigma_i,
        mask=mask, reduction=reduction,
    )


def loss_phys_from_model(
    model: torch.nn.Module, # Expects PINNLevitator
    t: torch.Tensor,
    u: torch.Tensor,
    m: float = 0.018,      # kg
    g: float = 9.81,       # m/s^2
    R: float = 2.72,       # Ohm
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate the physics-based loss for the levitator system."""
    t.requires_grad_(True)
    y_hat, i_hat = model(t)

    # --- Calculate derivatives using Auto-Diff ---
    # First derivatives
    grad_outputs = torch.ones_like(y_hat)
    y_dot, = torch.autograd.grad(outputs=y_hat, inputs=t, grad_outputs=grad_outputs, create_graph=True)
    i_dot, = torch.autograd.grad(outputs=i_hat, inputs=t, grad_outputs=grad_outputs, create_graph=True)

    # Second derivative for position (acceleration)
    y_ddot, = torch.autograd.grad(outputs=y_dot, inputs=t, grad_outputs=grad_outputs, create_graph=True)

    # --- Get trainable physical parameters from model ---
    k0, k, a = model.get_phys_params()

    # --- Inductance and its derivative ---
    # L(y) = k0 + k / (1 + y/a)
    L_y = k0 + k / (1 + y_hat / a)
    # dL/dy = - (k * a) / (a + y)^2
    dL_dy = - (k * a) / ((a + y_hat) ** 2)

    # --- Physics Residuals ---
    # R_mech = m * y_ddot - (0.5 * dL_dy * i_hat^2 + m * g)
    f_mag = 0.5 * dL_dy * (i_hat ** 2)
    f_grav = m * g
    residual_mech = m * y_ddot - (f_mag + f_grav)

    # R_elect = L(y) * i_dot - (u - R * i_hat - dL_dy * y_dot * i_hat)
    v_ind = dL_dy * y_dot * i_hat
    residual_elect = L_y * i_dot - (u - R * i_hat - v_ind)

    # --- Loss Calculation ---
    loss_mech = torch.mean(residual_mech ** 2)
    loss_elect = torch.mean(residual_elect ** 2)
    loss = loss_mech + loss_elect

    if reduction == "mean":
        return loss
    elif reduction == "sum":
        return loss * t.shape[0]
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def get_residuals_from_model(
    model: torch.nn.Module, # Expects PINNLevitator
    t: torch.Tensor,
    u: torch.Tensor,
    m: float = 0.018,      # kg
    g: float = 9.81,       # m/s^2
    R: float = 2.72,       # Ohm
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate and return the physics-based residuals for the levitator system."""
    t.requires_grad_(True)
    y_hat, i_hat = model(t)

    # --- Calculate derivatives using Auto-Diff ---
    grad_outputs = torch.ones_like(y_hat)
    y_dot, = torch.autograd.grad(outputs=y_hat, inputs=t, grad_outputs=grad_outputs, create_graph=True)
    i_dot, = torch.autograd.grad(outputs=i_hat, inputs=t, grad_outputs=grad_outputs, create_graph=True)
    y_ddot, = torch.autograd.grad(outputs=y_dot, inputs=t, grad_outputs=grad_outputs, create_graph=True)

    # --- Get trainable physical parameters from model ---
    k0, k, a = model.get_phys_params()

    # --- Inductance and its derivative ---
    L_y = k0 + k / (1 + y_hat / a)
    dL_dy = - (k * a) / ((a + y_hat) ** 2)

    # --- Physics Residuals ---
    f_mag = 0.5 * dL_dy * (i_hat ** 2)
    f_grav = m * g
    residual_mech = m * y_ddot - (f_mag + f_grav)

    v_ind = dL_dy * y_dot * i_hat
    residual_elect = L_y * i_dot - (u - R * i_hat - v_ind)

    return residual_mech, residual_elect
