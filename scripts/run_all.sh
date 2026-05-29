#!/usr/bin/env bash
# scripts/run_all.sh  --  Full pipeline for HW4 on the GPU pod
# Run: bash scripts/run_all.sh

set -euo pipefail

cd /root/hw4

# 1. Dependencies
pip install -q torch-fidelity wandb huggingface_hub matplotlib tqdm
pip install -e .

# 2. Guided-diffusion setup
if [ ! -d "/root/guided-diffusion" ]; then
    git clone https://github.com/openai/guided-diffusion /root/guided-diffusion
    cd /root/guided-diffusion && pip install -e .
    cd /root/hw4
fi

mkdir -p /root/guided-diffusion/models
if [ ! -f "/root/guided-diffusion/models/256x256_diffusion_uncond.pt" ]; then
    wget -q -O /root/guided-diffusion/models/256x256_diffusion_uncond.pt \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
fi

# 3. Tests
python -m pytest tests/ -x -v

# 4. Train VP
python scripts/train_vp.py --epochs 50 --save_dir runs/vp --wandb

# 5. Train RectFlow
python scripts/train_rectflow.py --epochs 50 --save_dir runs/rectflow

# 6. Generate VP samples
mkdir -p runs/samples
for STEPS in 1 5 10 50 100 200 1000; do
    python scripts/sample.py --method em --checkpoint runs/vp/best.pt \
        --num_steps "$STEPS" --n_samples 64 \
        --out "runs/samples/em_steps${STEPS}.png"
done

for STEPS in 10 50 100 1000; do
    for CORR in 1 3; do
        python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \
            --num_steps "$STEPS" --n_corrector "$CORR" --n_samples 64 \
            --out "runs/samples/pc_steps${STEPS}_corr${CORR}.png"
    done
done

# 7. Generate RectFlow samples
for STEPS in 1 5 10 50 100 200; do
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow/best.pt \
        --num_steps "$STEPS" --n_samples 64 \
        --out "runs/samples/rf_steps${STEPS}.png"
done

# 8. Reflow
python scripts/train_rectflow.py --reflow \
    --checkpoint runs/rectflow/best.pt \
    --n_reflow_pairs 50000 --epochs 20 \
    --save_dir runs/rectflow_reflow

python scripts/sample.py --method rectflow \
    --checkpoint runs/rectflow_reflow/best.pt \
    --num_steps 1 --n_samples 64 \
    --out runs/samples/reflow_steps1.png

# 9. Comparison grid
python scripts/sample.py --method all \
    --vp_checkpoint runs/vp/best.pt \
    --rf_checkpoint runs/rectflow/best.pt \
    --reflow_checkpoint runs/rectflow_reflow/best.pt \
    --seed 42 --out runs/samples/comparison_grid.png

# 10. KID evaluation
python scripts/eval_kid.py \
    --vp_checkpoint runs/vp/best.pt \
    --rf_checkpoint runs/rectflow/best.pt \
    --reflow_checkpoint runs/rectflow_reflow/best.pt \
    --n_samples 1000 --device cuda --wandb

# 11. Coefficient plot (Problem 1.8)
python scripts/plot_coefficient.py --out runs/coefficient_plot.png

# 12. Push checkpoints to HF
python scripts/push_to_hf.py

echo "All done!"
