"""
scripts/sample.py  --  Generate and compare samples (Parts 5C, 6B, 6D)

Usage::
    # EM samples  (5.C.iii)
    python scripts/sample.py --method em --checkpoint runs/vp/best.pt \
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000

    # PC samples  (5.C.iv)
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 1

    # Rectified Flow Euler  (6.B)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow/best.pt \
        --num_steps 100

    # One-step reflow  (6.C)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow_reflow/best.pt \
        --num_steps 1

    # Side-by-side grid  (6.D)
    python scripts/sample.py --method all --vp_checkpoint runs/vp/best.pt \
        --rf_checkpoint runs/rectflow/best.pt \
        --reflow_checkpoint runs/rectflow_reflow/best.pt \
        --seed 42 --out comparison_grid.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


def save_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a (B,1,H,W) tensor as an image grid."""
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow)
    plt.figure(figsize=(nrow, samples.size(0) // nrow + 1))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",      type=str, default="em",
                   choices=["em", "pc", "rectflow", "all"])
    p.add_argument("--checkpoint",          type=str, default=None)
    p.add_argument("--vp_checkpoint",       type=str, default=None)
    p.add_argument("--rf_checkpoint",       type=str, default=None)
    p.add_argument("--reflow_checkpoint",   type=str, default=None)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--num_steps",   type=int,   default=1000)
    p.add_argument("--n_corrector", type=int,   default=1)
    p.add_argument("--snr",         type=float, default=0.16)
    p.add_argument("--n_samples",   type=int,   default=64)
    p.add_argument("--out",    type=str, default="samples.png")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _load_vp(checkpoint: str, args, device):
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return sde, model


def _load_rf(checkpoint: str, device):
    flow = RectifiedFlow()
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return flow, model


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape = (args.n_samples, 1, 28, 28)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.method == "em":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = _load_vp(ckpt, args, device)
        samples = sde.euler_maruyama(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out, title=f"EM steps={args.num_steps}")

    elif args.method == "pc":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = _load_vp(ckpt, args, device)
        samples = sde.predictor_corrector(
            model, shape,
            num_steps=args.num_steps,
            n_corrector=args.n_corrector,
            snr=args.snr,
            device=device,
        )
        save_grid(samples, args.out, title=f"PC steps={args.num_steps} corr={args.n_corrector}")

    elif args.method == "rectflow":
        ckpt = args.checkpoint or args.rf_checkpoint
        flow, model = _load_rf(ckpt, device)
        samples = flow.euler_sample(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out, title=f"RectFlow Euler steps={args.num_steps}")

    elif args.method == "all":
        n = 8
        shape8 = (n, 1, 28, 28)
        rows = []

        # Row 1: VP EM
        sde, vp_model = _load_vp(args.vp_checkpoint, args, device)
        rows.append(sde.euler_maruyama(vp_model, shape8, num_steps=args.num_steps, device=device))

        # Row 2: VP PC
        rows.append(sde.predictor_corrector(
            vp_model, shape8, num_steps=args.num_steps,
            n_corrector=args.n_corrector, snr=args.snr, device=device,
        ))

        # Row 3: RectFlow
        flow, rf_model = _load_rf(args.rf_checkpoint, device)
        rows.append(flow.euler_sample(rf_model, shape8, num_steps=100, device=device))

        # Row 4: Reflow 1-step
        flow2, reflow_model = _load_rf(args.reflow_checkpoint, device)
        rows.append(flow2.euler_sample(reflow_model, shape8, num_steps=1, device=device))

        grid_rows = []
        row_labels = ["VP EM", "VP PC", "RectFlow", "Reflow 1-step"]
        for label, row in zip(row_labels, rows):
            imgs = (row.clamp(-1, 1) * 0.5 + 0.5).cpu()
            grid_rows.append(make_grid(imgs, nrow=n))

        fig, axes = plt.subplots(len(rows), 1, figsize=(n * 1.5, len(rows) * 2))
        for ax, grid, label in zip(axes, grid_rows, row_labels):
            ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
            ax.set_ylabel(label, fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        plt.close()
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
