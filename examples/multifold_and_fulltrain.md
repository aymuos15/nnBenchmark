# Multi-Fold Training and Full Dataset Training

This guide covers two advanced training workflows: cross-validation with multiple folds and training on all available data for final models.

## Multi-Fold Cross-Validation (5-Fold CV)

Train on all 5 cross-validation folds to get robust performance estimates.

### Step 1: Generate Configs for All Folds

```bash
# Generate fold 0
nnBench.plan --dataset Dataset001_Hippo --fold 0 --verbose

# Generate fold 1
nnBench.plan --dataset Dataset001_Hippo --fold 1 --verbose

# Generate fold 2
nnBench.plan --dataset Dataset001_Hippo --fold 2 --verbose

# Generate fold 3
nnBench.plan --dataset Dataset001_Hippo --fold 3 --verbose

# Generate fold 4
nnBench.plan --dataset Dataset001_Hippo --fold 4 --verbose
```

**Or use a loop:**

```bash
for fold in {0..4}; do
  nnBench.plan --dataset Dataset001_Hippo --fold $fold --verbose
done
```

**Expected outputs:**
- `nnBench_results/Dataset001_Hippo/fold_0/fold_0.yaml`
- `nnBench_results/Dataset001_Hippo/fold_1/fold_1.yaml`
- `nnBench_results/Dataset001_Hippo/fold_2/fold_2.yaml`
- `nnBench_results/Dataset001_Hippo/fold_3/fold_3.yaml`
- `nnBench_results/Dataset001_Hippo/fold_4/fold_4.yaml`

### Step 2: Train All Folds

```bash
# Train each fold individually
for fold in {0..4}; do
  echo "Training fold $fold..."
  nnBench.train --config fold_${fold}.yaml --dataset Dataset001_Hippo
done
```

**Expected outputs per fold:**
- `nnBench_results/Dataset001_Hippo/fold_<N>/checkpoints/`
- `nnBench_results/Dataset001_Hippo/fold_<N>/training_history.json`

### Step 3: Run Inference on All Folds

```bash
for fold in {0..4}; do
  echo "Running inference fold $fold..."
  nnBench.inference --config fold_${fold}.yaml --dataset Dataset001_Hippo
done
```

**Expected outputs per fold:**
- `nnBench_results/Dataset001_Hippo/fold_<N>/predictions/`
- `nnBench_results/Dataset001_Hippo/fold_<N>/test_history.json`

### Step 4: Aggregate Results

```bash
# Create a results summary directory
mkdir -p fold_results

for fold in {0..4}; do
  cp nnBench_results/Dataset001_Hippo/fold_${fold}/test_history.json fold_results/fold_${fold}_results.json
done

# Optionally generate plots for each fold
for fold in {0..4}; do
  echo "Plotting fold $fold..."
  nnBench.plot --config fold_${fold}.yaml --dataset Dataset001_Hippo
done
```

### Complete Multi-Fold Bash Script

Save this as `run_5fold_cv.sh`:

```bash
#!/bin/bash
set -e

DATASET="Dataset001_Hippo"
NUM_FOLDS=5

echo "=== Multi-Fold Cross-Validation ==="
echo "Dataset: $DATASET"
echo ""

# Step 1: Plan all folds
echo "Step 1: Generating configs for all folds..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
  echo "  Planning fold $fold..."
  nnBench.plan --dataset $DATASET --fold $fold --verbose
done
echo "✓ Config generation complete"
echo ""

# Step 2: Train all folds
echo "Step 2: Training all folds..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
  echo "  Training fold $fold..."
  nnBench.train --config fold_${fold}.yaml --dataset $DATASET
  echo "  ✓ Fold $fold training complete"
done
echo "✓ Training complete"
echo ""

# Step 3: Run inference
echo "Step 3: Running inference on all folds..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
  echo "  Inference fold $fold..."
  nnBench.inference --config fold_${fold}.yaml --dataset $DATASET
  echo "  ✓ Fold $fold inference complete"
done
echo "✓ Inference complete"
echo ""

# Step 4: Generate plots
echo "Step 4: Generating plots..."
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
  echo "  Plotting fold $fold..."
  nnBench.plot --config fold_${fold}.yaml --dataset $DATASET
done
echo "✓ All plots generated"
echo ""

echo "=== 5-Fold CV Complete ==="
echo "Results available in: nnBench_results/$DATASET/"
```

Run with:
```bash
chmod +x run_5fold_cv.sh
./run_5fold_cv.sh
```

---

## Full Dataset Training (fold: -1)

Train on all available data without a validation split. Use this for the final model when you want to leverage all data for maximum performance.

### Generate Config for Full Dataset

```bash
# Generate config for fold -1 (all data)
nnBench.plan --dataset Dataset001_Hippo --fold -1 --verbose
```

**Expected output:**
- `nnBench_results/Dataset001_Hippo/fold_-1/fold_-1.yaml`

### Train on Full Dataset

```bash
nnBench.train --config fold_-1.yaml --dataset Dataset001_Hippo
```

**Key differences from fold 0-4:**
- No validation split (all training data used for training)
- No validation metrics in training history
- Checkpoint saved for final epoch (not selected by best metric)
- `training_history.json` contains only training loss, no validation metrics

**Expected outputs:**
- `nnBench_results/Dataset001_Hippo/fold_-1/checkpoints/`
- `nnBench_results/Dataset001_Hippo/fold_-1/training_history.json`

### Run Inference (Optional)

```bash
# Note: Without a validation split, there's no labeled test set
# You can run inference on unlabeled data if available
nnBench.inference --config fold_-1.yaml --dataset Dataset001_Hippo
```

### Config Structure for fold: -1

In the generated `fold_-1.yaml`:

```yaml
dataset:
  fold: -1        # Special value indicating full dataset training
  num_classes: 2
  patch_size: [128, 128, 128]
  # ... other settings

training:
  epochs: 200
  batch_size: 2
  # No validation split, so val_interval is ignored
  # ...
```

---

## Comparison: 5-Fold CV vs Full Dataset Training

| Aspect | 5-Fold CV | Full Dataset (fold: -1) |
|--------|-----------|------------------------|
| **When to Use** | Evaluate model robustness, get performance estimate | Final model for deployment |
| **Training Data** | 80% per fold | 100% of training data |
| **Validation** | Yes (20% per fold) | No |
| **Metrics** | Per-fold results, cross-fold mean ± std | Training metrics only |
| **Reproducibility** | Deterministic (seed=12345) | Deterministic (seed=12345) |
| **Training Time** | 5× single fold | Similar to one fold |
| **Use Case** | Research, benchmarking | Production deployment |

---

## Tips and Best Practices

### Parallelizing Fold Training

To speed up training, you can run multiple folds in parallel on different GPUs:

```bash
# Train fold 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 nnBench.train --config fold_0.yaml --dataset Dataset001_Hippo &

# Train fold 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 nnBench.train --config fold_1.yaml --dataset Dataset001_Hippo &

# Wait for all to complete
wait
echo "All folds training complete"
```

### Monitor Training Progress

```bash
# Watch training in real-time
watch -n 5 'tail -20 nnBench_results/Dataset001_Hippo/fold_0/training_history.json'

# Or use tail -f
tail -f nnBench_results/Dataset001_Hippo/fold_0/training_history.json
```

### Resuming Interrupted Training

```bash
# Resume from checkpoint if training was interrupted
nnBench.train --config fold_0.yaml --dataset Dataset001_Hippo --continue
```

### Generate Summary of All Folds

Create a simple Python script to aggregate results:

```python
import json
from pathlib import Path

dataset = "Dataset001_Hippo"
results_dir = Path("nnBench_results") / dataset

# Collect dice scores from all folds
all_fold_results = []
for fold in range(5):
    test_history_path = results_dir / f"fold_{fold}" / "test_history.json"
    if test_history_path.exists():
        with open(test_history_path) as f:
            data = json.load(f)
            dice = data["summary"]["DiceMetric"]["mean"]
            all_fold_results.append({
                "fold": fold,
                "dice": dice
            })

# Print summary
print(f"Dataset: {dataset}")
print(f"Fold-wise Dice Scores:")
for result in all_fold_results:
    print(f"  Fold {result['fold']}: {result['dice']:.4f}")

mean_dice = sum(r["dice"] for r in all_fold_results) / len(all_fold_results)
print(f"\nMean Dice (5-fold CV): {mean_dice:.4f}")
```

---

## Troubleshooting

**Q: My training was interrupted. How do I resume?**

A: Use the `--continue` flag:
```bash
nnBench.train --config fold_0.yaml --dataset Dataset001_Hippo --continue
```

**Q: Can I train multiple folds in parallel?**

A: Yes! Use different GPUs or batch jobs. Just ensure each has a unique GPU assigned via `CUDA_VISIBLE_DEVICES`.

**Q: What's the difference between fold 0 and fold -1?**

A: Fold 0 trains on 80% of data with 20% validation split. Fold -1 trains on 100% of data with no validation.

**Q: Should I use 5-fold CV or full dataset training?**

A: Use 5-fold CV for research and evaluation (get confidence estimates). Use fold -1 for your final production model to leverage all available data.

