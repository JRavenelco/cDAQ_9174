from __future__ import annotations

import torch
import torch.nn as nn


class TimeMLP(nn.Module):
    def __init__(self, y_mu: float, y_std: float, i_mu: float, i_std: float,
                 hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = []
        in_features = 1
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.Tanh())
            in_features = hidden
        layers.append(nn.Linear(in_features, 2))  # outputs: [y_hat_norm, i_hat_norm]
        self.net = nn.Sequential(*layers)

        # Buffers for de-normalization
        self.register_buffer('y_mu',  torch.tensor(float(y_mu)))
        self.register_buffer('y_std', torch.tensor(float(y_std)))
        self.register_buffer('i_mu',  torch.tensor(float(i_mu)))
        self.register_buffer('i_std', torch.tensor(float(i_std)))

    @torch.no_grad()
    def set_stats(self, y_mu: float, y_std: float, i_mu: float, i_std: float):
        self.y_mu.fill_(float(y_mu))
        self.y_std.fill_(float(y_std))
        self.i_mu.fill_(float(i_mu))
        self.i_std.fill_(float(i_std))

    def forward(self, t_norm: torch.Tensor):
        z = self.net(t_norm)
        y_hat_norm = z[:, :1]
        i_hat_norm = z[:, 1:]
        y_hat = self.y_mu + self.y_std * y_hat_norm
        i_hat = self.i_mu + self.i_std * i_hat_norm
        return y_hat, i_hat


class PINNLevitator(torch.nn.Module):
    """Container model that holds both the TimeMLP and trainable physical parameters."""
    def __init__(self, y_mu, y_std, i_mu, i_std, hidden, depth):
        super().__init__()
        self.mlp = TimeMLP(y_mu, y_std, i_mu, i_std, hidden, depth)

        # Store log of parameters to ensure they remain positive during optimization
        self.log_k0 = torch.nn.Parameter(torch.log(torch.tensor(36.3e-3)))
        self.log_k = torch.nn.Parameter(torch.log(torch.tensor(3.5e-3)))
        self.log_a = torch.nn.Parameter(torch.log(torch.tensor(5.2e-3)))

    def forward(self, t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Passes the input through the MLP."""
        return self.mlp(t_norm)

    def get_phys_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the positive physical parameters from their log storage."""
        k0 = torch.exp(self.log_k0)
        k = torch.exp(self.log_k)
        a = torch.exp(self.log_a)
        return k0, k, a
