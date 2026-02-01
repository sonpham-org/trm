# Kaggle Experiment Plan - Week of Feb 3, 2026

## Objective
Test two hypotheses to optimize TRM training efficiency:
1. **Model Size Hypothesis**: Can we achieve competitive performance with 50% fewer parameters?
2. **Augmentation Hypothesis**: Can we maintain performance with 50% less augmentation?

Both experiments target **50% reduction in training time** while minimizing performance loss.

---

## Experiment Setup

### Baseline (for comparison)
- **Model**: 7M parameters (hidden_size=512, 8 heads, 2 layers)
- **Dataset**: arc1concept-aug-1000 (~400 base puzzles × 1000 aug)
- **Expected Training Time**: ~72 hours on 4× H100 → **~15-18 hours on Kaggle TPU v3-8**
- **Expected Performance**: 45% on ARC-AGI-1, 8% on ARC-AGI-2

---

## Experiment 1: Half Model Size

### Goal
Determine if 3.5M parameter model can achieve >40% on ARC-AGI-1 with half the training time.

### Configuration
**File**: `config/arch/trm_small.yaml` (already created)

**Key Changes**:
```yaml
hidden_size: 362      # Was 512 (50% reduction in params)
num_heads: 6          # Was 8 (proportional reduction)
expansion: 4          # Keep same
L_layers: 2           # Keep same
H_cycles: 3           # Keep same
L_cycles: 6           # Keep same
```

**Expected Parameters**: ~3.5M (50% of 7M)

### Training Command
```bash
# On Kaggle GPU P100 (recommended for this size)
run_name="trm_small_arc1_kaggle"
python pretrain.py \
arch=trm_small \
data_paths="[/kaggle/input/arc-dataset/arc1concept-aug-1000]" \
evaluators="[arc@ARCEvaluator]" \
eval_data_dir=/kaggle/input/arc-dataset/arc1concept-aug-1000/test \
epochs=50000 eval_interval=2500 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=512 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

### Expected Results
- **Training Time**: ~7-9 hours on Kaggle TPU
- **Target Performance**: 40-43% on ARC-AGI-1 (vs 45% baseline)
- **Verdict**: If >40%, half-size model is viable for faster iteration!

---

## Experiment 2: Reduced Augmentation

### Goal
Test if 500× augmentation (vs 1000×) maintains performance while halving training time.

### Dataset Preparation

#### Step 1: Create Reduced Augmentation Dataset
```bash
# Run this locally or on Kaggle
python -m dataset.build_arc_dataset \
  --input-file-prefix /kaggle/input/arc-raw/combined/arc-agi \
  --output-dir data/arc1concept-aug-500 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 500
```

**Key Change**: `--num-aug 500` (was 1000)

#### Step 2: Upload to Kaggle as Dataset
- Create new Kaggle Dataset: "arc1concept-aug-500"
- Upload the generated `data/arc1concept-aug-500/` folder
- Make it public or accessible to your notebooks

### Training Command
```bash
# On Kaggle TPU v3-8
run_name="trm_baseline_arc1_aug500"
python pretrain.py \
arch=trm \
data_paths="[/kaggle/input/arc1concept-aug-500]" \
evaluators="[arc@ARCEvaluator]" \
eval_data_dir=/kaggle/input/arc1concept-aug-500/test \
epochs=50000 eval_interval=2500 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
global_batch_size=768 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

### Expected Results
- **Training Time**: ~7-9 hours on Kaggle TPU (vs 15-18h baseline)
- **Target Performance**: 42-45% on ARC-AGI-1 (minimal degradation)
- **Verdict**: If >42%, we can halve training time without significant loss!

---

## Experiment Schedule (One Week)

### Your Account:

| Day | Resource | Experiment | Duration | Notes |
|-----|----------|------------|----------|-------|
| **Mon** | GPU P100 | Exp 1: Small Model | 8h | Test model size reduction |
| **Tue** | Setup | Create aug-500 dataset | 2h | Prepare for Exp 2 |
| **Wed-Thu** | TPU v3-8 | Exp 2: Reduced Aug | 8h | Test augmentation reduction |
| **Fri** | Analysis | Compare results | 2h | Evaluate both experiments |

**Total GPU**: 8h / 30h quota (27% used)  
**Total TPU**: 8h / 30h quota (27% used)

### Friend's Account (Optional - Double Speed):

| Day | Resource | Experiment | Duration | Notes |
|-----|----------|------------|----------|-------|
| **Mon** | TPU v3-8 | Baseline (full 7M) | 16h | Reference comparison |
| **Tue-Wed** | GPU P100 | Exp 1 Replica | 8h | Verify small model results |
| **Thu-Fri** | Analysis | - | - | Help analyze & plan next steps |

**Combined**: 4 experiments in 5 days! 🚀

---

## Success Metrics

### Experiment 1 (Small Model)
- ✅ **Success**: >40% on ARC-AGI-1 (within 5% of baseline)
- ⚠️ **Partial**: 35-40% (consider 25% smaller model instead)
- ❌ **Failure**: <35% (stick with 7M baseline)

### Experiment 2 (Reduced Aug)
- ✅ **Success**: >42% on ARC-AGI-1 (within 3% of baseline)
- ⚠️ **Partial**: 38-42% (try 750× aug as compromise)
- ❌ **Failure**: <38% (1000× aug is necessary)

---

## What to Track

### During Training (Wandb):
```python
# Key metrics to log
- train/loss
- train/accuracy
- eval/arc_accuracy (ARC-AGI-1)
- eval/arc2_accuracy (if testing on ARC-AGI-2)
- train/halt_steps (average number of reasoning iterations)
- time/epoch_time
- time/total_time
```

### After Training:
- Final checkpoint accuracy
- Training time (total GPU/TPU hours)
- Memory usage (max VRAM/HBM)
- Convergence speed (epochs to reach 90% of final performance)

---

## Decision Tree (Post-Experiments)

```
Both experiments successful (>40% and >42%)?
├─ YES → Use small model + reduced aug for 4x faster iteration! 🎉
│        Next: Combine both (3.5M params + 500 aug) for potential 75% time reduction
│
└─ NO → Which failed?
   ├─ Small model only → Use 7M + reduced aug (2x speedup)
   ├─ Reduced aug only → Use small model + 1000 aug (2x speedup)
   └─ Both failed → Keep baseline, focus on other improvements
```

---

## Backup Plans

### If Quota Runs Out:
- Download checkpoints from intermediate steps
- Resume training next week from last checkpoint
- Use friend's account to continue

### If Training Fails:
- Reduce batch size: `global_batch_size=384` (instead of 512/768)
- Enable gradient checkpointing (if OOM)
- Switch between GPU and TPU based on availability

### If Results Are Inconclusive:
- Run a third experiment: 25% smaller model (hidden_size=443)
- Try 750× augmentation as middle ground
- Test on Sudoku first (faster iteration)

---

## Pre-Flight Checklist

### Before Starting Experiments:

- [ ] **Kaggle Account**: Phone verified, GPU/TPU access confirmed
- [ ] **Datasets Uploaded**: 
  - [ ] arc1concept-aug-1000 (baseline)
  - [ ] arc1concept-aug-500 (reduced aug)
- [ ] **Wandb Setup**: 
  - [ ] API key configured
  - [ ] Project created: "trm-kaggle-experiments"
- [ ] **Code Ready**:
  - [ ] trm_small.yaml config verified
  - [ ] pretrain.py paths updated for Kaggle
  - [ ] Checkpoint save frequency set to 500 steps
- [ ] **Test Run**: 
  - [ ] 1000-step test completed successfully
  - [ ] No errors in data loading or model initialization

---

## Code Modifications for Kaggle

### 1. Update Data Paths in Notebook
```python
# Cell 1: Setup paths
import os
os.environ['WANDB_API_KEY'] = 'your-key-here'

DATA_BASE = "/kaggle/input"
OUTPUT_BASE = "/kaggle/working"

# Modify pretrain.py paths (or use command-line args)
```

### 2. Enable Frequent Checkpointing
```bash
# Add to training command
checkpoint_interval=500  # Save every 500 steps (vs default 2500)
```

### 3. Auto-Download Best Checkpoint
```python
# Cell (end of notebook): Download best model
import shutil
from pathlib import Path

# Find best checkpoint by eval metric
checkpoints = sorted(Path(OUTPUT_BASE).glob("*/checkpoint_*.pt"))
best_ckpt = checkpoints[-1]  # or filter by eval metric

# Save to Kaggle output
shutil.copy(best_ckpt, "/kaggle/working/best_model.pt")
print(f"Best checkpoint saved: {best_ckpt}")
```

---

## Analysis Template (After Experiments)

### Results Summary Table:

| Experiment | Model Size | Aug | Training Time | ARC-AGI-1 | ARC-AGI-2 | Speedup | Verdict |
|------------|------------|-----|---------------|-----------|-----------|---------|---------|
| Baseline | 7M | 1000× | 15-18h | 45% | 8% | 1.0× | Reference |
| Exp 1 | 3.5M | 1000× | 7-9h | ?% | ?% | 2.0× | ? |
| Exp 2 | 7M | 500× | 7-9h | ?% | ?% | 2.0× | ? |
| Exp 3 (bonus) | 3.5M | 500× | 3-5h | ?% | ?% | 4.0× | ? |

### Questions to Answer:
1. **Does model size scale linearly with performance?** (If 3.5M → 40%, is 14M → 50%?)
2. **Is augmentation the bottleneck?** (If 500× works, try even less?)
3. **What's the next bottleneck?** (After halving time, what slows us down?)

---

## Next Steps (Week 2)

### If Experiments Succeed:
1. **Combine winners**: Test small model + reduced aug together
2. **Scale up winners**: If 500× works, test with 300× aug
3. **New architectures**: Try 3-layer model (L_layers=3)

### If Experiments Fail:
1. **Partial success**: Use whichever worked
2. **Focus shift**: Investigate other improvements (test-time training, better optimizer)
3. **Baseline validation**: Reproduce 45% result to ensure setup is correct

---

## Resources

- **Kaggle Documentation**: https://www.kaggle.com/docs/notebooks
- **TPU Training Guide**: https://www.kaggle.com/docs/tpu
- **Wandb for Kaggle**: https://docs.wandb.ai/guides/integrations/kaggle
- **Your Codebase**: /home/son/Desktop/GitHub/TinyRecursiveModels/
- **TRM Paper**: https://arxiv.org/abs/2510.04871

---

## Emergency Contacts

- **Kaggle Support**: https://www.kaggle.com/contact
- **Wandb Support**: support@wandb.com
- **Friend's Contact**: [Your friend's contact info]

---

## Notes Section

### Experiment 1 Notes (Small Model):
```
[To be filled after experiment]
- Started: [Date/Time]
- Ended: [Date/Time]
- Final Accuracy: 
- Observations:
- Issues Encountered:
```

### Experiment 2 Notes (Reduced Aug):
```
[To be filled after experiment]
- Started: [Date/Time]
- Ended: [Date/Time]
- Final Accuracy:
- Observations:
- Issues Encountered:
```

---

**Good luck with your Kaggle experiments! May both hypotheses prove successful and unlock 4× faster iteration! 🚀**
