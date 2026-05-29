"""
scripts/eval_kid.py  --  KID evaluation for VP and RectFlow models

Compute KID (Kernel Inception Distance) for each method and step count.
Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \
        --vp_checkpoint  runs/vp/best.pt \
        --rf_checkpoint  runs/rectflow/best.pt \
        --beta_min 0.01 --beta_max 5.0 \
        --n_samples 1000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError("torch-fidelity is required: pip install torch-fidelity")

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint",     type=str, required=True)
    p.add_argument("--rf_checkpoint",     type=str, required=True)
    p.add_argument("--reflow_checkpoint", type=str, default=None)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb",     action="store_true")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str, subset_size: int = 1000) -> dict:
    """Compute KID between generated and real image directories."""
    n_gen = len([f for f in os.listdir(generated_dir) if f.endswith(".png")])
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(subset_size, n_gen),
        verbose=False,
    )
    return metrics


def save_real_images(real_dir: str, n: int = 1000):
    """Save real FashionMNIST images to a directory for KID comparison."""
    os.makedirs(real_dir, exist_ok=True)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    for i in range(min(n, len(ds))):
        img, _ = ds[i]
        img_01 = (img.clamp(-1, 1) * 0.5 + 0.5)
        save_image(img_01, os.path.join(real_dir, f"{i:05d}.png"))


def main():
    args = get_args()
    device = torch.device(args.device)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity="dhruvsheth", project="cs148b-hw4", name="eval_kid")

    # Load models
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()

    rf_flow = RectifiedFlow()
    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()

    reflow_model = None
    if args.reflow_checkpoint:
        reflow_model = UNet(in_channels=1, base_channels=64).to(device)
        reflow_model.load_state_dict(torch.load(args.reflow_checkpoint, map_location=device))
        reflow_model.eval()

    # Save real images once
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        save_real_images(real_dir, n=args.n_samples)

        shape = (args.n_samples, 1, 28, 28)
        results = {}

        methods = [("em", sde, vp_model, "EM"), ("rf", rf_flow, rf_model, "RectFlow")]
        if reflow_model is not None:
            methods.append(("reflow", rf_flow, reflow_model, "Reflow"))

        header = f"{'Method':<12} " + " ".join(f"steps={s:<5}" for s in STEP_COUNTS)
        print(header)
        print("-" * len(header))

        for key, model_or_sde, model, label in methods:
            row_vals = []
            for steps in STEP_COUNTS:
                gen_dir = os.path.join(tmpdir, f"{key}_{steps}")
                with torch.no_grad():
                    if key == "em":
                        # cap at 1000 for VP EM
                        actual_steps = min(steps, 1000)
                        samples = model_or_sde.euler_maruyama(
                            model, shape, num_steps=actual_steps, device=device
                        )
                    else:
                        samples = rf_flow.euler_sample(
                            model, shape, num_steps=steps, device=device
                        )
                save_samples_to_dir(samples, gen_dir)
                metrics = compute_kid(gen_dir, real_dir, subset_size=args.n_samples)
                kid_mean = metrics.get("kernel_inception_distance_mean", float("nan"))
                kid_std  = metrics.get("kernel_inception_distance_std",  float("nan"))
                row_vals.append(f"{kid_mean:.4f}+-{kid_std:.4f}")
                if run:
                    run.log({f"kid/{label}_steps{steps}": kid_mean})
            print(f"{label:<12} " + " ".join(f"{v:<12}" for v in row_vals))
            results[label] = row_vals

    if run:
        run.finish()


if __name__ == "__main__":
    main()
