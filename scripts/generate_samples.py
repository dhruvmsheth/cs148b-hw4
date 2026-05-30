"""Generate sample grids from trained VP, RectFlow, and Reflow models."""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


def load_model(ckpt_path, device):
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def generate_vp_samples(model, sde, n=32, steps=1000, method="em", device="cuda"):
    shape = (n, 1, 28, 28)
    if method == "em":
        return sde.euler_maruyama(model, shape, num_steps=steps, device=device)
    else:
        return sde.predictor_corrector(model, shape, num_steps=steps, device=device)


def generate_rf_samples(model, flow, n=32, steps=100, device="cuda"):
    shape = (n, 1, 28, 28)
    return flow.euler_sample(model, shape, num_steps=steps, device=device)


def save_grid(samples, path, nrow=8):
    samples = samples.clamp(-1, 1) * 0.5 + 0.5
    grid = make_grid(samples, nrow=nrow, padding=2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_training_curves(loss_paths, labels, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for path, label in zip(loss_paths, labels):
        losses = np.load(path)
        ax.plot(range(1, len(losses) + 1), losses, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vp-ckpt", type=Path, default=Path("runs/vp/best.pt"))
    p.add_argument("--rf-ckpt", type=Path, default=Path("runs/rectflow/best.pt"))
    p.add_argument("--reflow-ckpt", type=Path, default=Path("runs/rectflow_reflow/best.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("figures"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    sde = VPSDE()
    flow = RectifiedFlow()

    # VP samples
    if args.vp_ckpt.exists():
        print("Generating VP-SDE samples...")
        vp_model = load_model(args.vp_ckpt, device)
        em_samples = generate_vp_samples(vp_model, sde, n=32, steps=1000, method="em", device=device)
        save_grid(em_samples, args.output_dir / "vp_em_samples.pdf")
        pc_samples = generate_vp_samples(vp_model, sde, n=32, steps=200, method="pc", device=device)
        save_grid(pc_samples, args.output_dir / "vp_pc_samples.pdf")

    # RectFlow samples
    if args.rf_ckpt.exists():
        print("Generating RectFlow samples...")
        rf_model = load_model(args.rf_ckpt, device)
        rf_samples = generate_rf_samples(rf_model, flow, n=32, steps=100, device=device)
        save_grid(rf_samples, args.output_dir / "rectflow_samples.pdf")

    # Reflow samples
    if args.reflow_ckpt.exists():
        print("Generating Reflow samples...")
        reflow_model = load_model(args.reflow_ckpt, device)
        reflow_1step = generate_rf_samples(reflow_model, flow, n=32, steps=1, device=device)
        save_grid(reflow_1step, args.output_dir / "reflow_1step_samples.pdf")
        reflow_samples = generate_rf_samples(reflow_model, flow, n=32, steps=10, device=device)
        save_grid(reflow_samples, args.output_dir / "reflow_10step_samples.pdf")

    # Comparison grid: 4 rows x 8 cols
    if all(p.exists() for p in [args.vp_ckpt, args.rf_ckpt, args.reflow_ckpt]):
        print("Generating comparison grid...")
        vp_model = load_model(args.vp_ckpt, device)
        rf_model = load_model(args.rf_ckpt, device)
        reflow_model = load_model(args.reflow_ckpt, device)

        em_8 = generate_vp_samples(vp_model, sde, n=8, steps=1000, method="em", device=device)
        pc_8 = generate_vp_samples(vp_model, sde, n=8, steps=200, method="pc", device=device)
        rf_8 = generate_rf_samples(rf_model, flow, n=8, steps=100, device=device)
        reflow_8 = generate_rf_samples(reflow_model, flow, n=8, steps=1, device=device)

        all_samples = torch.cat([em_8, pc_8, rf_8, reflow_8], dim=0)
        all_samples = all_samples.clamp(-1, 1) * 0.5 + 0.5
        grid = make_grid(all_samples, nrow=8, padding=2)

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        ax.axis("off")
        row_labels = ["VP EM (1000)", "VP PC (200)", "RectFlow (100)", "Reflow (1)"]
        for i, label in enumerate(row_labels):
            ax.text(-5, 30 * i + 15, label, fontsize=10, va="center", ha="right")
        fig.savefig(args.output_dir / "comparison_grid.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {args.output_dir / 'comparison_grid.pdf'}")

    # Training curves
    vp_train = Path("runs/vp/train_losses.npy")
    rf_train = Path("runs/rectflow/train_losses.npy")
    reflow_train = Path("runs/rectflow_reflow/train_losses.npy")

    if vp_train.exists():
        vp_val = Path("runs/vp/val_losses.npy")
        paths = [vp_train]
        labels = ["VP train"]
        if vp_val.exists():
            paths.append(vp_val)
            labels.append("VP val")
        plot_training_curves(paths, labels, args.output_dir / "vp_training_curves.pdf")

    if rf_train.exists():
        paths = [rf_train]
        labels = ["RectFlow"]
        if reflow_train.exists():
            paths.append(reflow_train)
            labels.append("Reflow")
        plot_training_curves(paths, labels, args.output_dir / "rectflow_training_curves.pdf")

    # Dataset visualization
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
    data_samples = torch.stack([ds[i][0] for i in range(32)])
    save_grid(data_samples, args.output_dir / "fashionmnist_samples.pdf")


if __name__ == "__main__":
    main()
