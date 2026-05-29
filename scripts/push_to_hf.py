"""scripts/push_to_hf.py  --  Upload checkpoints and samples to HuggingFace."""

from __future__ import annotations

import os
from huggingface_hub import HfApi

REPO_ID = "dhruvmsheth/cs148b-hw4-diffusion"

FILES = [
    ("runs/vp/best.pt",               "vp_best.pt"),
    ("runs/rectflow/best.pt",          "rectflow_best.pt"),
    ("runs/rectflow_reflow/best.pt",   "rectflow_reflow_best.pt"),
    ("runs/vp/train_losses.npy",       "vp_train_losses.npy"),
    ("runs/vp/val_losses.npy",         "vp_val_losses.npy"),
    ("runs/rectflow/train_losses.npy", "rectflow_train_losses.npy"),
    ("runs/coefficient_plot.png",      "coefficient_plot.png"),
    ("runs/samples/comparison_grid.png", "comparison_grid.png"),
]


def main():
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)

    for local_path, repo_name in FILES:
        if os.path.exists(local_path):
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_name,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"Uploaded: {repo_name}")
        else:
            print(f"Skipping missing: {local_path}")

    # Upload all sample images
    samples_dir = "runs/samples"
    if os.path.isdir(samples_dir):
        api.upload_folder(
            folder_path=samples_dir,
            path_in_repo="samples",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("Uploaded samples/ folder")


if __name__ == "__main__":
    main()
