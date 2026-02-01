# Kaggle Notebook: Experiment 1 - Half Model Size

## Setup

This notebook tests a 3.5M parameter TRM model (50% reduction from 7M baseline).

### Environment Setup

```python
# Install dependencies
!pip install --quiet adam-atan2 hydra-core wandb

# Clone your repository (or upload code as Kaggle dataset)
!git clone https://github.com/YOUR_USERNAME/TinyRecursiveModels.git
%cd TinyRecursiveModels

# Wandb login
import wandb
wandb.login(key='YOUR_WANDB_KEY')
```

### Verify TPU/GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch version: {torch.__version__}")
```

---

## Quick Test Run (1000 steps)

```bash
%%bash
# 5-minute test to verify setup
run_name="test_small_model"
python pretrain.py \
arch=trm_small \
data_paths="[/kaggle/input/arc1concept-aug-1000]" \
evaluators="[]" \
epochs=1000 eval_interval=500 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=512 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

**Expected output**: Training should start, loss should decrease. Check for any errors.

---

## Full Training Run

```bash
%%bash
# Full training: ~7-9 hours on P100 GPU
run_name="exp1_small_model_arc1_kaggle"
python pretrain.py \
arch=trm_small \
data_paths="[/kaggle/input/arc1concept-aug-1000]" \
evaluators="[arc@ARCEvaluator]" \
eval_data_dir=/kaggle/input/arc1concept-aug-1000/test \
epochs=50000 eval_interval=2500 min_eval_interval=10000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=512 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True \
checkpoint_interval=500
```

---

## Monitor Progress

```python
# Check latest Wandb run
import wandb
api = wandb.Api()
runs = api.runs("your-username/trm-kaggle-experiments")
latest_run = runs[0]

print(f"Run: {latest_run.name}")
print(f"State: {latest_run.state}")
print(f"Runtime: {latest_run.summary.get('_runtime', 0) / 3600:.1f} hours")

# Plot training curve
import pandas as pd
history = latest_run.history()
history[['train/loss', 'eval/arc_accuracy']].plot(figsize=(12, 5))
```

---

## Download Results

```python
# Find best checkpoint
from pathlib import Path
import shutil

checkpoints = sorted(Path("outputs").rglob("checkpoint_*.pt"))
if checkpoints:
    latest_ckpt = checkpoints[-1]
    shutil.copy(latest_ckpt, "/kaggle/working/exp1_best_model.pt")
    print(f"✅ Checkpoint saved to /kaggle/working/exp1_best_model.pt")
    print(f"Size: {latest_ckpt.stat().st_size / 1e6:.1f} MB")
else:
    print("⚠️ No checkpoints found")
```

---

## Results Summary

```python
# Print final results
print("=" * 60)
print("EXPERIMENT 1 RESULTS: Half Model Size (3.5M params)")
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
    
    print(f"\nComparison to Baseline (7M params, 45%):")
    print(f"  Difference: {diff:+.1%}")
    
    if current_acc >= 0.40:
        print(f"  Verdict: ✅ SUCCESS - Within 5% of baseline!")
    elif current_acc >= 0.35:
        print(f"  Verdict: ⚠️ PARTIAL - Consider 25% reduction instead")
    else:
        print(f"  Verdict: ❌ FAILURE - Stick with 7M baseline")

print("=" * 60)
```

---

## Next Steps

**If successful (>40%):**
- Upload checkpoint as Kaggle Dataset
- Test on ARC-AGI-2 evaluation set
- Combine with Experiment 2 (reduced augmentation)

**If partial (35-40%):**
- Try 25% smaller model (hidden_size=443)
- Analyze where model struggles (visualize failure cases)

**If failed (<35%):**
- Keep 7M baseline for future experiments
- Focus on other improvements (architecture, training strategy)
