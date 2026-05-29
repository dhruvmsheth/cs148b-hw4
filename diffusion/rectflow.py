"""
diffusion/rectflow.py  --  Rectified Flow
Part 6 of EE/CS 148B HW4.

Reference: Liu et al. (2023) "Flow Straight and Fast: Learning to Generate
and Transfer Data with Rectified Flow" (ICLR 2023).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RectifiedFlow:
    """Rectified Flow forward process, training loss, and ODE sampler.

    Interpolation: X_t = (1-t) X_0 + t X_1, X_0 ~ N(0,I), X_1 ~ data.
    """

    def __init__(self) -> None:
        pass

    def forward_process(
        self, x1: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample from the rectified flow interpolation at time t.

        Returns (x_t, x0, vel) where vel = x1 - x0.
        """
        x0 = torch.randn_like(x1)
        t_broad = t.view(-1, *([1] * (x1.dim() - 1)))
        x_t = (1.0 - t_broad) * x0 + t_broad * x1
        vel = x1 - x0
        return x_t, x0, vel

    def loss(self, v_theta: nn.Module, x1: Tensor) -> Tensor:
        """Rectified Flow MSE loss: E[||(X1-X0) - v_theta(X_t, t)||^2]."""
        t = torch.rand(x1.shape[0], device=x1.device)
        x_t, _, vel = self.forward_process(x1, t)
        pred = v_theta(x_t, t)
        return F.mse_loss(pred, vel)

    # ------------------------------------------------------------------
    # Euler ODE sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_sample(
        self,
        v_theta: nn.Module,
        shape: tuple[int, ...],
        num_steps: int = 100,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Euler ODE sampler integrating dX/dt = v_theta(X_t, t) from t=0 to t=1."""
        dt = 1.0 / num_steps
        x = torch.randn(shape, device=device)
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full((shape[0],), t_val, device=device)
            v = v_theta(x, t)
            x = x + v * dt
        return x

    # ------------------------------------------------------------------
    # Reflow pair generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        v_theta: nn.Module,
        n_pairs: int,
        image_shape: tuple[int, ...],
        num_steps: int = 100,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Generate (X_0, X_1) pairs by running the Euler ODE on fresh noise.

        Returns tensors of shape (n_pairs, C, H, W) on CPU.
        """
        x0_list, x1_list = [], []
        remaining = n_pairs
        while remaining > 0:
            bs = min(batch_size, remaining)
            x0_batch = torch.randn(bs, *image_shape, device=device)
            dt = 1.0 / num_steps
            x = x0_batch.clone()
            for i in range(num_steps):
                t_val = i * dt
                t = torch.full((bs,), t_val, device=device)
                v = v_theta(x, t)
                x = x + v * dt
            x0_list.append(x0_batch.cpu())
            x1_list.append(x.cpu())
            remaining -= bs
        return torch.cat(x0_list, dim=0), torch.cat(x1_list, dim=0)
