"""
scripts/plot_coefficient.py  --  Part 1.8
Plot the DDPM loss coefficient beta_t^2 / (2 * sigma_t^2 * alpha_t * (1 - alpha_bar_t))
vs. t on a log-scale y-axis.

Usage::
    python scripts/plot_coefficient.py --T 1000 --beta_start 1e-4 --beta_end 0.02
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


def linear_schedule(T: int, beta_start: float, beta_end: float):
    """Return linear beta schedule of length T."""
    return np.linspace(beta_start, beta_end, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",          type=int,   default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end",   type=float, default=0.02)
    parser.add_argument("--out",        type=str,   default="coefficient_plot.png")
    args = parser.parse_args()

    betas = linear_schedule(args.T, args.beta_start, args.beta_end)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    # sigma_t^2 = 1 - alpha_bar_t (for the posterior variance approximation)
    # Here sigma_t^2 refers to the diffusion noise variance beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    # but Problem 1.8 asks for the simplified coefficient:
    # lambda_t = beta_t^2 / (2 * sigma_t^2 * alpha_t * (1 - alpha_bar_t))
    # where sigma_t^2 = beta_t (simplified, or posterior variance)
    # Using posterior variance: sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
    sigma_sq = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    coeff = betas ** 2 / (2.0 * sigma_sq * alphas * (1.0 - alpha_bars))
    t = np.arange(1, args.T + 1)

    plt.figure(figsize=(8, 4))
    plt.semilogy(t, coeff)
    plt.xlabel("t")
    plt.ylabel(r"$\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$")
    plt.title("DDPM Loss Coefficient vs. t (log scale)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
