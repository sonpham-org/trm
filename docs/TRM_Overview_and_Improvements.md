# Tiny Recursive Models (TRM): Overview and Improvement Strategies

## 1. What is Tiny Recursive Models (TRM)?

### Background

**Tiny Recursive Model (TRM)** is a groundbreaking recursive reasoning approach introduced in the paper "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871) by Alexia Jolicoeur-Martineau in October 2025. TRM achieves remarkable results on hard reasoning tasks like ARC-AGI with a tiny 7M parameter neural network, demonstrating that massive foundational models are not always necessary for complex reasoning tasks.

### Key Results

- **45% accuracy on ARC-AGI-1** with only 7M parameters
- **8% accuracy on ARC-AGI-2** with the same tiny model
- Outperforms most Large Language Models (LLMs) including Deepseek R1, o3-mini, and Gemini 2.5 Pro
- Uses **less than 0.01% of the parameters** compared to these LLMs
- Also achieves strong results on Sudoku-Extreme (~87%) and Maze-Hard tasks

### Core Philosophy

TRM challenges the prevailing notion that bigger models are always better. Instead, it shows that:
- **"Less is More"**: Small models with recursive reasoning can solve hard problems
- **Parameter Efficiency**: 7M parameters vs. billions in LLMs
- **Simplicity**: Strips away complexity from previous approaches like Hierarchical Reasoning Model (HRM)
- **No biological justification needed**: Pure algorithmic efficiency without appealing to brain analogies

### How TRM Works

TRM is based on a simple but powerful recursive reasoning loop:

```
Input: Question x, Initial answer y₀, Initial latent z₀

For each improvement step k (up to K steps):
    1. Recursive Reasoning Phase (n iterations):
       - Update latent: z ← f(x, y, z)  [repeat n times]
    
    2. Answer Update Phase:
       - Update answer: y ← g(y, z)

Output: Final answer y_K
```

**Key Components:**

1. **Question Encoding (x)**: Embedded representation of the input puzzle/problem
2. **Answer State (y)**: Current predicted answer that gets progressively refined
3. **Latent State (z)**: Hidden reasoning state that captures intermediate reasoning steps
4. **Two Update Functions**:
   - `f(x, y, z)`: Updates latent state given question, current answer, and current latent
   - `g(y, z)`: Updates answer given current answer and latent state

### Architecture Details

**Network Structure:**
- **Tiny Size**: Only 7M parameters total
- **2-Layer Transformer** (or 2-layer MLP variant)
- **No Hierarchy**: Unlike HRM, uses a single network (not high/low frequency components)
- **Recursive Cycles**: 
  - H_cycles: Number of answer improvement iterations (typically 3)
  - L_cycles: Number of latent update iterations per improvement (typically 4-6)

**Two Variants:**

1. **TRM with Attention** (`pretrain_att_*`):
   - Uses standard self-attention transformer blocks
   - Better for spatial reasoning tasks (ARC, Maze)
   
2. **TRM with MLP** (`pretrain_mlp_t_*`):
   - Uses MLP operating on sequence dimension instead of attention
   - More parameter efficient
   - Better for constrained problems (Sudoku-Extreme: 87% vs 75%)

### Training Approach

**Data Efficiency:**
- Trained on only ~1000 base examples
- Heavy data augmentation (1000 augmentations per example)
- Total effective training data: ~1M examples

**Training Hyperparameters:**
- Learning rate: 1e-4 (both model and puzzle embeddings)
- Weight decay: 1.0 for Sudoku, 0.1 for ARC
- Optimizer: AdamAtan2 (β₁=0.9, β₂=0.95)
- Warmup steps: 2000
- Epochs: 50,000 for Sudoku/Maze, 100,000 for ARC
- Batch size: 768 (global)
- Optional EMA (Exponential Moving Average) with rate 0.999

**Key Features:**
- **Puzzle-specific embeddings**: Learned per-puzzle embeddings for better generalization
- **EMA**: Exponential moving average of weights for more stable predictions
- **Gradient accumulation**: For training with limited GPU memory
- **Distributed training**: Supports multi-GPU with PyTorch DDP

### Why TRM is Better than HRM

TRM simplifies Hierarchical Reasoning Model (HRM) in several ways:

1. **Single Network**: HRM uses two networks (high/low frequency), TRM uses one
2. **No Biological Constraints**: HRM appeals to brain-inspired hierarchies, TRM is purely algorithmic
3. **No Fixed-Point Theorems**: TRM doesn't require mathematical guarantees about convergence
4. **Simpler Training**: Fewer hyperparameters and architectural choices
5. **Better Generalization**: Despite simplicity, achieves higher accuracy on test sets

---

## 2. Potential Improvements and Research Directions

Based on the current implementation and recent advances in deep learning, here are potential improvement strategies:

### A. Training Strategy Improvements

#### 1. **Advanced Learning Rate Schedules**
- **Current**: Linear warmup + cosine decay
- **Improvements**:
  - Cyclic learning rates for better exploration
  - OneCycle policy for faster convergence
  - Layer-wise learning rate adaptation (LLRD)
  - Separate schedules for reasoning components (L vs H cycles)

#### 2. **Curriculum Learning**
- Start with easier puzzles, gradually increase difficulty
- Begin with fewer recursive cycles, increase during training
- Multi-stage training: pretrain on simple tasks, fine-tune on complex ones
- Difficulty-based sampling strategies

#### 3. **Improved Regularization**
- **Dropout variants**: DropBlock, DropPath for transformers
- **Stochastic depth**: Randomly skip layers during training
- **Mixup/CutMix**: For puzzle augmentation
- **Label smoothing**: Reduce overconfidence
- **Gradient clipping**: Current implementation may benefit from adaptive clipping

#### 4. **Data Augmentation Enhancements**
- **Current**: 1000 augmentations per example
- **Improvements**:
  - More sophisticated augmentation strategies (rotation, reflection, color permutation)
  - Adversarial augmentation: create hard examples
  - Mixup at the latent level
  - Temporal augmentation: varying sequence lengths

#### 5. **Multi-Task Learning**
- Train on multiple puzzle types simultaneously
- Shared backbone, task-specific heads
- Transfer learning from easier to harder tasks
- Auxiliary tasks for better representations

#### 6. **Better Optimization**
- **Alternative Optimizers**:
  - AdamW with scheduled weight decay
  - Lion optimizer (more memory efficient)
  - Sophia (second-order information)
  - LAMB for large batch training
- **Gradient Accumulation**: More stable with larger effective batch sizes
- **Mixed Precision Training**: Better utilize GPU memory and speed

### B. Architecture Improvements

#### 7. **Adaptive Computation Time (ACT)**
- **Current**: Fixed H_cycles and L_cycles
- **Improvement**: Learn when to stop reasoning
  - Halting mechanism based on confidence
  - Different puzzles need different reasoning depths
  - Q-learning for halting (partially implemented)
  - Reduce computation on easy examples

#### 8. **Attention Mechanisms**
- **Current**: Standard self-attention
- **Improvements**:
  - Flash Attention 2/3 for efficiency
  - Sliding window attention for long sequences
  - Cross-attention between question and answer
  - Memory-augmented attention
  - Sparse attention patterns

#### 9. **Positional Encodings**
- **Current**: RoPE (Rotary Position Embeddings) or none
- **Alternatives**:
  - ALiBi (Attention with Linear Biases)
  - Learned position embeddings
  - 2D position embeddings for grid-based puzzles
  - Relative position encodings

#### 10. **Recursive Reasoning Variants**
- **Bidirectional reasoning**: Forward and backward passes
- **Multi-scale reasoning**: Different cycle frequencies
- **Residual connections** across cycles
- **Gating mechanisms**: Learn what to update at each step
- **Ensemble of reasoners**: Multiple parallel reasoning paths

#### 11. **Memory Mechanisms**
- External memory banks for storing intermediate reasoning
- Neural Turing Machines / Differentiable Neural Computers
- Key-value memory for pattern matching
- Working memory buffers

### C. Training Hyperparameter Tuning

#### 12. **Critical Hyperparameters to Explore**

**Model Architecture:**
- `L_layers`: Try 3-4 layers instead of 2
- `H_cycles`: Test 2-5 (current: 3)
- `L_cycles`: Test 3-8 (current: 4-6)
- `hidden_size`: Current values work, but try 256, 384, 512
- `num_heads`: For attention variant, try 4, 8, 16
- `expansion`: MLP expansion ratio (current: likely 4), try 2-8

**Optimization:**
- `lr`: Current 1e-4 is conservative, try 5e-5 to 5e-4
- `puzzle_emb_lr`: Current 1e-4 for ARC, 1e-2 for others - explore 1e-5 to 1e-2
- `weight_decay`: Try 0.01, 0.05, 0.1, 0.5, 1.0
- `lr_warmup_steps`: Try 500, 1000, 5000
- `beta2`: Try 0.98, 0.999 for more stable updates

**Training:**
- `global_batch_size`: Try 256, 512, 1024, 1536
- `ema_rate`: Try 0.99, 0.995, 0.9999
- Different epoch counts with early stopping

#### 13. **Hyperparameter Search Strategies**
- Random search for initial exploration
- Bayesian optimization (Optuna, Weights & Biases Sweeps)
- Population-based training (PBT)
- Grid search for final refinement

### D. Checkpoint Strategies

#### 14. **Model Averaging and Ensembling**
- **Weight Averaging** (Stochastic Weight Averaging - SWA):
  - Average weights from different checkpoints
  - Exponential moving average (EMA) already implemented
  - Polyak averaging
  - Model soups: average multiple fine-tuned models

- **Checkpoint Ensembling**:
  - Test-time ensemble: average predictions from multiple checkpoints
  - Snapshot ensembling: save checkpoints at different learning rate cycle peaks
  - Horizontal ensembling: different random seeds

- **Best Checkpoint Selection**:
  - Not just last checkpoint
  - Checkpoint with best validation accuracy
  - Average of top-k checkpoints
  - Staged evaluation on validation set

#### 15. **Better Checkpointing Strategies**
- Save more frequent checkpoints during critical training phases
- Keep checkpoints from different learning rate schedules
- Version control for checkpoints with metadata
- Checkpoint pruning strategies to save disk space

### E. Evaluation and Analysis Improvements

#### 16. **Better Metrics**
- Not just exact accuracy
- Partial credit scoring
- Confidence calibration metrics
- Per-puzzle-type analysis
- Error analysis: what types of puzzles fail?

#### 17. **Interpretability**
- Visualize latent state evolution across cycles
- Attention map visualization
- Track which puzzles benefit from more cycles
- Analyze learned puzzle embeddings

### F. Advanced Techniques

#### 18. **Self-Supervised Pretraining**
- Pretrain on unlabeled puzzle data
- Contrastive learning on puzzle representations
- Masked prediction tasks
- Puzzle reconstruction

#### 19. **Meta-Learning**
- Learn to learn new puzzle types quickly
- MAML or Reptile for fast adaptation
- Few-shot learning on new puzzle categories

#### 20. **Knowledge Distillation**
- Distill from larger models
- Self-distillation from EMA model
- Student-teacher frameworks

---

## 3. Practical Next Steps

### Immediate Improvements (Low Hanging Fruit)

1. **Implement Better Checkpoint Selection**:
   - Currently uses latest checkpoint
   - Should compare: latest, best validation, EMA, average of top-3

2. **Hyperparameter Tuning**:
   - Run grid search on H_cycles (2, 3, 4, 5)
   - Run grid search on L_cycles (3, 4, 5, 6, 7, 8)
   - Test different learning rates

3. **Improve Data Augmentation**:
   - Analyze which augmentations help most
   - Add new augmentation strategies

4. **Implement Adaptive Computation**:
   - Use the existing ACT (Adaptive Computation Time) infrastructure
   - Learn to halt early on easy examples

5. **Better Regularization**:
   - Add dropout to prevent overfitting
   - Experiment with stochastic depth

### Medium-Term Improvements

1. **Architecture Search**:
   - Try different layer counts
   - Experiment with hybrid attention/MLP architectures
   - Test memory-augmented variants

2. **Multi-Task Training**:
   - Train on Sudoku + Maze + ARC simultaneously
   - Transfer learning experiments

3. **Advanced Ensembling**:
   - Implement checkpoint averaging strategies
   - Snapshot ensembling during training

### Long-Term Research Directions

1. **Scaling Laws**:
   - Understand how performance scales with:
     - Model size
     - Dataset size
     - Recursive cycles
     - Training compute

2. **Theoretical Understanding**:
   - Why does recursive reasoning work?
   - Convergence guarantees
   - Relationship to iterative refinement algorithms

3. **New Applications**:
   - Apply to other reasoning domains
   - Mathematical theorem proving
   - Code synthesis
   - Scientific reasoning

---

## 4. Recommended Training Configuration Changes

### For Better Generalization

```yaml
# Conservative (better generalization)
lr: 5e-5  # Lower learning rate
weight_decay: 0.5  # More regularization
ema: True
ema_rate: 0.999  # Slower EMA
global_batch_size: 1024  # Larger batches

# Add dropout
dropout: 0.1

# More cycles for harder problems
arch.H_cycles: 4
arch.L_cycles: 6
```

### For Faster Training

```yaml
# Aggressive (faster convergence)
lr: 2e-4
lr_warmup_steps: 1000
weight_decay: 0.1
global_batch_size: 512

# Fewer cycles initially
arch.H_cycles: 2
arch.L_cycles: 4
```

### For Maximum Performance (Recommended)

```yaml
# Balanced approach
lr: 1e-4
puzzle_emb_lr: 5e-4  # Higher for embeddings
weight_decay: 0.3
puzzle_emb_weight_decay: 0.3

global_batch_size: 768
beta2: 0.98  # Slightly lower for more adaptive

ema: True
ema_rate: 0.999

# Architecture
arch.L_layers: 3  # One more layer
arch.H_cycles: 4  # One more cycle
arch.L_cycles: 6

# Training schedule
epochs: 75000
lr_warmup_steps: 3000
checkpoint_every_n_steps: 1000  # More frequent checkpoints
```

---

## References

1. Jolicoeur-Martineau, A. (2025). "Less is More: Recursive Reasoning with Tiny Networks". arXiv:2510.04871
2. Wang, G., et al. (2025). "Hierarchical Reasoning Model". arXiv:2506.21734
3. Current codebase: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
