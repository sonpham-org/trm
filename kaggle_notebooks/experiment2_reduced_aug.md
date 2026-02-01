# Kaggle Notebook: Experiment 2 - Reduced Augmentation

## Setup

This notebook tests 500× augmentation (50% reduction from 1000× baseline).

### Environment Setup

```python
# Install dependencies
!pip install --quiet adam-atan2 hydra-core wandb

# Clone your repository
!git clone https://github.com/YOUR_USERNAME/TinyRecursiveModels.git
%cd TinyRecursiveModels

# Wandb login
import wandb
wandb.login(key='YOUR_WANDB_KEY')
```

---

## Build Reduced Augmentation Dataset

**Option A: Build on Kaggle (if you have raw ARC data uploaded)**

```bash
%%bash
# Build dataset with 500× augmentation
python -m dataset.build_arc_dataset \
  --input-file-prefix /kaggle/input/arc-raw/combined/arc-agi \
  --output-dir /kaggle/working/arc1concept-aug-500 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 500

# Check dataset size
du -sh /kaggle/working/arc1concept-aug-500
```

**Option B: Use Pre-built Dataset (recommended)**

If you built the dataset locally and uploaded it as a Kaggle Dataset:

```python
# Verify dataset is accessible
import os
from pathlib import Path

dataset_path = Path("/kaggle/input/arc1concept-aug-500")
assert dataset_path.exists(), "❌ Dataset not found! Upload arc1concept-aug-500 first"

print("✅ Dataset found!")
print(f"Contents: {list(dataset_path.iterdir())}")
```

---

## Quick Test Run (1000 steps)

```bash
%%bash
# 5-minute test
run_name="test_reduced_aug"
python pretrain.py \
arch=trm \
data_paths="[/kaggle/input/arc1concept-aug-500]" \
evaluators="[]" \
epochs=1000 eval_interval=500 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=768 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

---

## Full Training Run (TPU)

**Important**: Use TPU for this experiment (better for full 7M model).

```bash
%%bash
# Full training: ~7-9 hours on TPU v3-8
run_name="exp2_reduced_aug_arc1_kaggle"

# Configure TPU (if using TPU runtime)
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

python pretrain.py \
arch=trm \
data_paths="[/kaggle/input/arc1concept-aug-500]" \
evaluators="[arc@ARCEvaluator]" \
eval_data_dir=/kaggle/input/arc1concept-aug-500/test \
epochs=50000 eval_interval=2500 min_eval_interval=10000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=768 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True \
checkpoint_interval=500
```

**Note**: Kaggle TPU setup may require additional configuration. If TPU doesn't work, fall back to GPU with `global_batch_size=512`.

---

## Monitor Progress

```python
# Real-time monitoring
import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("your-username/trm-kaggle-experiments")
latest_run = runs[0]

print(f"Run: {latest_run.name}")
print(f"State: {latest_run.state}")
print(f"Runtime: {latest_run.summary.get('_runtime', 0) / 3600:.1f} hours")

# Plot metrics
history = latest_run.history()
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss curve
history['train/loss'].plot(ax=axes[0], title='Training Loss')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Step')

# Accuracy curve (if available)
if 'eval/arc_accuracy' in history.columns:
    history['eval/arc_accuracy'].plot(ax=axes[1], title='ARC-AGI-1 Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Step')
    axes[1].axhline(y=0.45, color='r', linestyle='--', label='Baseline (1000× aug)')
    axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## Compare Augmentation Levels

```python
# Compare training dynamics between aug levels
import json
from pathlib import Path

# If you have metadata from both datasets
datasets = {
    '1000× aug': '/kaggle/input/arc1concept-aug-1000/train/dataset.json',
    '500× aug': '/kaggle/input/arc1concept-aug-500/train/dataset.json'
}

print("Dataset Comparison:")
print("=" * 60)
for name, path in datasets.items():
    if Path(path).exists():
        with open(path) as f:
            meta = json.load(f)
        print(f"{name}:")
        print(f"  Total puzzles: {meta['total_puzzles']}")
        print(f"  Mean examples/puzzle: {meta['mean_puzzle_examples']:.1f}")
        print(f"  Est. total examples: ~{int(meta['total_puzzles'] * meta['mean_puzzle_examples'])}")
        print()
```

---

## Download Results

```python
# Save best checkpoint
from pathlib import Path
import shutil

checkpoints = sorted(Path("outputs").rglob("checkpoint_*.pt"))
if checkpoints:
    latest_ckpt = checkpoints[-1]
    shutil.copy(latest_ckpt, "/kaggle/working/exp2_best_model.pt")
    print(f"✅ Checkpoint saved to /kaggle/working/exp2_best_model.pt")
    print(f"Size: {latest_ckpt.stat().st_size / 1e6:.1f} MB")
else:
    print("⚠️ No checkpoints found")
```

---

## Results Summary

```python
# Print final results
print("=" * 60)
print("EXPERIMENT 2 RESULTS: Reduced Augmentation (500×)")
print("=" * 60)

if 'latest_run' in locals():
    summary = latest_run.summary
    print(f"Training Time: {summary.get('_runtime', 0) / 3600:.1f} hours")
    print(f"Final Train Loss: {summary.get('train/loss', 'N/A'):.4f}")
    print(f"ARC-AGI-1 Accuracy: {summary.get('eval/arc_accuracy', 'N/A'):.1%}")
    print(f"Average Halt Steps: {summary.get('train/halt_steps', 'N/A'):.2f}")
    
    # Compare to baseline
    baseline_acc = 0.45
    current_acc = summary.get('eval/arc_accuracy', 0)
    diff = current_acc - baseline_acc
    
    print(f"\nComparison to Baseline (1000× aug, 45%):")
    print(f"  Difference: {diff:+.1%}")
    
    if current_acc >= 0.42:
        print(f"  Verdict: ✅ SUCCESS - Minimal degradation!")
        print(f"           Can halve training time with reduced aug!")
    elif current_acc >= 0.38:
        print(f"  Verdict: ⚠️ PARTIAL - Try 750× aug as compromise")
    else:
        print(f"  Verdict: ❌ FAILURE - 1000× augmentation necessary")

print("=" * 60)
```

---

## Analyze Failure Cases

```python
# If you want to understand where reduced aug struggles
# (Requires loading model and running inference)

import torch
from pathlib import Path

# Load model checkpoint
ckpt_path = "/kaggle/working/exp2_best_model.pt"
if Path(ckpt_path).exists():
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print("Checkpoint info:")
    print(f"  Step: {checkpoint.get('step', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Model state dict keys
    model_keys = list(checkpoint.get('model_state_dict', {}).keys())
    print(f"  Model parameters: {len(model_keys)}")
else:
    print("No checkpoint found for analysis")
```

---

## Next Steps

**If successful (>42%):**
- Test on ARC-AGI-2 to verify generalization
- Try even less augmentation (300×) for 3× speedup
- Combine with Experiment 1 (small model + reduced aug)

**If partial (38-42%):**
- Build intermediate dataset: arc1concept-aug-750
- Analyze which puzzle types need more augmentation

**If failed (<38%):**
- Keep 1000× augmentation
- Focus on better augmentation strategies (semantic vs geometric)
- Investigate curriculum learning (start with heavy aug, reduce over time)

---

## Experiment 3 Bonus: Combined (Small + Reduced Aug)

**If both Exp 1 and Exp 2 succeed, run this for 4× speedup!**

```bash
%%bash
# 3.5M model + 500× aug = ~3-5 hours training!
run_name="exp3_small_reduced_arc1_kaggle"
python pretrain.py \
arch=trm_small \
data_paths="[/kaggle/input/arc1concept-aug-500]" \
evaluators="[arc@ARCEvaluator]" \
eval_data_dir=/kaggle/input/arc1concept-aug-500/test \
epochs=50000 eval_interval=2500 min_eval_interval=10000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=512 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

**Target**: >38% accuracy with 4× faster training would be a massive win!
