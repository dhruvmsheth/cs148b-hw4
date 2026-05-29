"""
diffusion/vp.py  --  Variance-Preserving (VP) SDE
Part 5 of EE/CS 148B HW4.

Reference: Song et al. (2021) "Score-Based Generative Modeling through
Stochastic Differential Equations" (Song21), Appendix B & D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VPSDE:
    """Variance-Preserving SDE forward process and samplers.

    The VP-SDE is: dx = -1/2 * beta(t) * x dt + sqrt(beta(t)) dB_t
    with beta(t) = beta_min + (beta_max - beta_min) * t (linear schedule).
    """

    def __init__(self, beta_min: float = 0.01, beta_max: float = 5.0, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t: Tensor) -> Tensor:
        """beta(t) -- the linear noise schedule."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def c(self, t: Tensor) -> Tensor:
        """c(t) = exp(-1/2 * integral_0^t beta(s) ds) -- signal decay factor."""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def sigma(self, t: Tensor) -> Tensor:
        """sigma(t) = sqrt(1 - c(t)^2) -- noise standard deviation."""
        return torch.sqrt(torch.clamp(1.0 - self.c(t) ** 2, min=0.0))

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Drift coefficient f(x, t) = -1/2 * beta(t) * x."""
        b = self.beta(t)
        # broadcast to spatial dims
        while b.dim() < x.dim():
            b = b.unsqueeze(-1)
        return -0.5 * b * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Diffusion coefficient g(t) = sqrt(beta(t))."""
        return torch.sqrt(self.beta(t))

    def marginal(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Sample from the forward marginal q(x_t | x_0).

        x_t = c(t) * x_0 + sigma(t) * eps, eps ~ N(0, I).
        """
        c_t = self.c(t)
        s_t = self.sigma(t)
        while c_t.dim() < x0.dim():
            c_t = c_t.unsqueeze(-1)
            s_t = s_t.unsqueeze(-1)
        eps = torch.randn_like(x0)
        x_t = c_t * x0 + s_t * eps
        return x_t, eps

    def score(self, eps_model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        """Convert eps prediction to score: s = -eps_theta(x,t) / sigma(t)."""
        s_t = self.sigma(t)
        while s_t.dim() < x.dim():
            s_t = s_t.unsqueeze(-1)
        eps = eps_model(x, t)
        return -eps / (s_t + 1e-8)

    # ------------------------------------------------------------------
    # Samplers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_maruyama(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Euler-Maruyama reverse-SDE sampler.

        Integrates the reverse VP-SDE from t=1 to t~=0.
        """
        num_steps = num_steps or self.T
        dt = 1.0 / num_steps

        t_init = torch.ones(shape[0], device=device)
        sigma_T = self.sigma(t_init)
        while sigma_T.dim() < len(shape):
            sigma_T = sigma_T.unsqueeze(-1)
        x = sigma_T * torch.randn(shape, device=device)

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((shape[0],), t_val, device=device)
            score = self.score(score_model, x, t)

            b = self.beta(t)
            while b.dim() < x.dim():
                b = b.unsqueeze(-1)

            drift = -b * (0.5 * x + score)
            diffusion = torch.sqrt(b) * torch.randn_like(x)
            x = x - drift * dt + diffusion * (dt ** 0.5)

        return x

    @torch.no_grad()
    def predictor_corrector(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        n_corrector: int = 1,
        snr: float = 0.16,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Predictor-Corrector sampler with EM predictor.

        Follows Algorithm 5 of Song21.
        """
        num_steps = num_steps or self.T
        dt = 1.0 / num_steps

        t_init = torch.ones(shape[0], device=device)
        sigma_T = self.sigma(t_init)
        while sigma_T.dim() < len(shape):
            sigma_T = sigma_T.unsqueeze(-1)
        x = sigma_T * torch.randn(shape, device=device)

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((shape[0],), t_val, device=device)

            # Predictor: EM step
            score = self.score(score_model, x, t)
            b = self.beta(t)
            while b.dim() < x.dim():
                b = b.unsqueeze(-1)
            drift = -b * (0.5 * x + score)
            x = x - drift * dt + torch.sqrt(b) * (dt ** 0.5) * torch.randn_like(x)

            # Corrector: annealed Langevin dynamics
            t_next = max(t_val - dt, 1e-5)
            t_c = torch.full((shape[0],), t_next, device=device)
            for _ in range(n_corrector):
                score_c = self.score(score_model, x, t_c)
                noise = torch.randn_like(x)
                score_norm = score_c.view(shape[0], -1).norm(dim=-1).mean()
                noise_norm = noise.view(shape[0], -1).norm(dim=-1).mean()
                step_size = 2 * (snr * noise_norm / (score_norm + 1e-8)) ** 2
                x = x + step_size * score_c + (2 * step_size) ** 0.5 * noise

        return x

    # ------------------------------------------------------------------
    # Inverse problems (EC)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint(
        self,
        score_model: nn.Module,
        corrupted: Tensor,
        mask: Tensor,
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Conditional reverse diffusion for inpainting (EC Problem 5.D).

        Replaces known pixels at each step with their forward-diffused values.
        """
        num_steps = num_steps or self.T
        dt = 1.0 / num_steps
        shape = corrupted.shape
        corrupted = corrupted.to(device)
        mask = mask.to(device)

        t_init = torch.ones(shape[0], device=device)
        sigma_T = self.sigma(t_init)
        while sigma_T.dim() < len(shape):
            sigma_T = sigma_T.unsqueeze(-1)
        x = sigma_T * torch.randn(shape, device=device)

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((shape[0],), t_val, device=device)

            # Forward diffuse the known pixels to time t
            x_known, _ = self.marginal(corrupted, t)

            score = self.score(score_model, x, t)
            b = self.beta(t)
            while b.dim() < x.dim():
                b = b.unsqueeze(-1)
            drift = -b * (0.5 * x + score)
            x_pred = x - drift * dt + torch.sqrt(b) * (dt ** 0.5) * torch.randn_like(x)

            # Replace observed pixels with forward-noised ground truth
            x = mask * x_known + (1 - mask) * x_pred

        return x
