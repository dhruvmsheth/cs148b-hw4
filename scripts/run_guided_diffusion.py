"""
scripts/run_guided_diffusion.py  --  Single-GPU sampling with guided-diffusion
Supports Problems 7.1 (unconditional), 7.2 (progressive), 7.3 (interpolation).
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '/root/guided-diffusion')
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


UNCOND_MODEL = '/root/guided-diffusion/models/256x256_diffusion_uncond.pt'


def load_model(model_path: str, device):
    """Load the 256x256 unconditional diffusion model."""
    defaults = model_and_diffusion_defaults()
    defaults.update({
        'image_size': 256,
        'num_channels': 256,
        'num_res_blocks': 2,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'attention_resolutions': '32,16,8',
        'dropout': 0.0,
        'diffusion_steps': 1000,
        'noise_schedule': 'linear',
        'timestep_respacing': '',
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
        'class_cond': False,
        'use_fp16': False,
        'use_new_attention_order': False,
        'learn_sigma': True,
        'resblock_updown': True,
    })
    model, diffusion = create_model_and_diffusion(**{
        k: v for k, v in defaults.items()
        if k in model_and_diffusion_defaults()
    })
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, diffusion


def sample_unconditional(model, diffusion, n: int, device, ddim: bool = False):
    """Sample n images unconditionally."""
    sample_fn = diffusion.ddim_sample_loop if ddim else diffusion.p_sample_loop
    samples = sample_fn(
        model,
        (n, 3, 256, 256),
        clip_denoised=True,
        model_kwargs={},
        device=device,
    )
    return ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['7_1', '7_2', '7_3'], default='7_1')
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default='runs/guided_diffusion')
    parser.add_argument('--ddim', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading model from {UNCOND_MODEL}...')
    model, diffusion = load_model(UNCOND_MODEL, device)

    if args.task == '7_1':
        print(f'Generating {args.n_samples} unconditional samples...')
        images = sample_unconditional(model, diffusion, args.n_samples, device, ddim=args.ddim)
        out_path = os.path.join(args.out_dir, f'samples_7_1_{images.shape[0]}x256x256x3.npz')
        np.savez(out_path, images)
        print(f'Saved: {out_path}')

    elif args.task == '7_2':
        # Progressive denoising: save intermediate steps
        print('Generating progressive denoising samples...')
        B = 1
        noise = torch.randn(B, 3, 256, 256, device=device)
        intermediates = []
        save_every = diffusion.num_timesteps // 8  # 8 frames

        # Collect intermediates via a custom callback
        collected = []
        def save_fn(img, i):
            if i % save_every == 0 or i == diffusion.num_timesteps - 1:
                arr = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                collected.append(arr[0])

        for i in range(diffusion.num_timesteps - 1, -1, -1):
            t = torch.tensor([i], device=device)
            with torch.no_grad():
                out = diffusion.p_mean_variance(model, noise, t, clip_denoised=True, model_kwargs={})
            noise_coeff = diffusion.sqrt_one_minus_alphas_cumprod[i]
            mean_coeff = diffusion.sqrt_alphas_cumprod[i]
            if i > 0:
                noise_next = torch.randn_like(noise)
                sigma = (diffusion.betas[i] * (1 - diffusion.alphas_cumprod_prev[i]) /
                         (1 - diffusion.alphas_cumprod[i])) ** 0.5
                noise = out["mean"] + sigma * noise_next
            else:
                noise = out["mean"]
            if i % save_every == 0 or i == 0:
                arr = ((noise + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                collected.append(arr[0])

        collected = collected[:8]  # Keep 8 frames
        for idx, frame in enumerate(collected):
            out_path = os.path.join(args.out_dir, f'progressive_{idx:02d}.npz')
            np.savez(out_path, np.expand_dims(frame, 0))
        print(f'Saved {len(collected)} progressive frames to {args.out_dir}')

    elif args.task == '7_3':
        # Noise interpolation between two noise vectors
        print('Generating noise interpolation samples...')
        z0 = torch.randn(1, 3, 256, 256, device=device)
        z1 = torch.randn(1, 3, 256, 256, device=device)
        alphas = torch.linspace(0, 1, 8, device=device)
        all_images = []
        for alpha in alphas:
            # Slerp interpolation
            z = (1 - alpha) * z0 + alpha * z1
            z = z / z.norm() * (z0.norm() * (1 - alpha) + z1.norm() * alpha)
            samples = diffusion.p_sample_loop(
                model, z.shape, noise=z,
                clip_denoised=True, model_kwargs={},
            )
            img = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            all_images.append(img[0])
        arr = np.stack(all_images, axis=0)
        out_path = os.path.join(args.out_dir, f'interpolation_8x256x256x3.npz')
        np.savez(out_path, arr)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
