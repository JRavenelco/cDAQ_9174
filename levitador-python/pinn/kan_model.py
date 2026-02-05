"""
Kolmogorov-Arnold Network (KAN) for PINN applications.
Uses learnable B-spline activation functions on edges instead of fixed activations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineActivation(nn.Module):
    """Learnable B-spline activation function for KAN edges."""
    
    def __init__(self, num_knots: int = 8, degree: int = 3):
        """
        Args:
            num_knots: Number of control points for the B-spline
            degree: Degree of the B-spline (3 = cubic)
        """
        super().__init__()
        self.num_knots = num_knots
        self.degree = degree
        
        # Learnable control points (coefficients)
        self.coefficients = nn.Parameter(torch.randn(num_knots))
        
        # Fixed knot vector (uniform spacing for simplicity)
        # Extended by degree on each side for proper B-spline behavior
        knots = torch.linspace(-1, 1, num_knots - degree + 1)
        knots = torch.cat([
            knots[0].repeat(degree),
            knots,
            knots[-1].repeat(degree)
        ])
        self.register_buffer('knots', knots)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline at input points.
        Args:
            x: Input tensor of any shape
        Returns:
            Output tensor of same shape as x
        """
        # Clamp input to knot range to avoid extrapolation issues
        x_clamped = torch.clamp(x, self.knots[self.degree], self.knots[-self.degree-1])
        
        # Evaluate B-spline basis functions using Cox-de Boor recursion
        # For efficiency, we use a simplified cubic B-spline approximation
        # Map x to [0, num_knots-1] range
        x_scaled = (x_clamped - self.knots[self.degree]) / (self.knots[-self.degree-1] - self.knots[self.degree])
        x_scaled = x_scaled * (self.num_knots - 1)
        
        # Get integer and fractional parts
        indices = torch.floor(x_scaled).long().clamp(0, self.num_knots - 2)
        t = x_scaled - indices.float()
        
        # Cubic B-spline blending (simplified for speed)
        # B0(t) = (1-t)^3 / 6
        # B1(t) = (3t^3 - 6t^2 + 4) / 6
        # B2(t) = (-3t^3 + 3t^2 + 3t + 1) / 6
        # B3(t) = t^3 / 6
        t2 = t * t
        t3 = t2 * t
        
        b0 = (1 - t)**3 / 6
        b1 = (3*t3 - 6*t2 + 4) / 6
        b2 = (-3*t3 + 3*t2 + 3*t + 1) / 6
        b3 = t3 / 6
        
        # Blend coefficients
        output = (
            b0 * self.coefficients[indices.clamp(0, self.num_knots-1)] +
            b1 * self.coefficients[(indices+1).clamp(0, self.num_knots-1)] +
            b2 * self.coefficients[(indices+2).clamp(0, self.num_knots-1)] +
            b3 * self.coefficients[(indices+3).clamp(0, self.num_knots-1)]
        )
        
        return output


class KANLayer(nn.Module):
    """Single KAN layer with learnable B-spline activations on edges."""
    
    def __init__(self, in_features: int, out_features: int, num_knots: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Each edge (in_i -> out_j) has its own learnable activation
        # This is the key difference from MLP: activation is on edges, not nodes
        self.activations = nn.ModuleList([
            nn.ModuleList([
                BSplineActivation(num_knots=num_knots)
                for _ in range(out_features)
            ])
            for _ in range(in_features)
        ])
        
        # Optional: add residual linear connection for stability
        self.residual = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_normal_(self.residual.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            (batch, out_features)
        """
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # For each input dimension
        for i in range(self.in_features):
            x_i = x[:, i:i+1]  # (batch, 1)
            
            # Apply learnable activation to each output connection
            for j in range(self.out_features):
                output[:, j] += self.activations[i][j](x_i).squeeze(-1)
        
        # Add residual connection for stability
        output = output + self.residual(x)
        
        return output


class KANLevitator(nn.Module):
    """
    KAN-based model for magnetic levitator.
    Replacement for TimeMLP with learnable activation functions.
    """
    
    def __init__(
        self,
        y_mu: float,
        y_std: float,
        i_mu: float,
        i_std: float,
        hidden: int = 64,
        depth: int = 3,
        num_knots: int = 8
    ):
        super().__init__()
        
        # Build KAN layers
        layers = []
        in_features = 1
        for _ in range(depth):
            layers.append(KANLayer(in_features, hidden, num_knots=num_knots))
            in_features = hidden
        
        # Final layer to 2 outputs (y, i)
        layers.append(KANLayer(in_features, 2, num_knots=num_knots))
        
        self.kan_net = nn.Sequential(*layers)
        
        # Normalization buffers (same as TimeMLP for compatibility)
        self.register_buffer('y_mu', torch.tensor(float(y_mu)))
        self.register_buffer('y_std', torch.tensor(float(y_std)))
        self.register_buffer('i_mu', torch.tensor(float(i_mu)))
        self.register_buffer('i_std', torch.tensor(float(i_std)))
        
        # Trainable physical parameters (log-space for positivity)
        self.log_k0 = nn.Parameter(torch.log(torch.tensor(36.3e-3)))
        self.log_k = nn.Parameter(torch.log(torch.tensor(3.5e-3)))
        self.log_a = nn.Parameter(torch.log(torch.tensor(5.2e-3)))
    
    def forward(self, t_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_norm: Normalized time (batch, 1)
        Returns:
            y_hat: Predicted position (batch, 1)
            i_hat: Predicted current (batch, 1)
        """
        z = self.kan_net(t_norm)  # (batch, 2)
        y_hat_norm = z[:, :1]
        i_hat_norm = z[:, 1:]
        
        # Denormalize
        y_hat = self.y_mu + self.y_std * y_hat_norm
        i_hat = self.i_mu + self.i_std * i_hat_norm
        
        return y_hat, i_hat
    
    def get_phys_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns positive physical parameters from log storage."""
        return torch.exp(self.log_k0), torch.exp(self.log_k), torch.exp(self.log_a)
