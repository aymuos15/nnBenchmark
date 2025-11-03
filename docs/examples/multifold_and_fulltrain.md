# Multi-Fold Cross-Validation and Full Dataset Training

## 5-Fold Cross-Validation

Generate and train all 5 folds:

```bash
# Generate all fold configs
for fold in {0..4}; do
  nnBench.plan --dataset Dataset001_Hippo --fold $fold
done

# Train all folds
for fold in {0..4}; do
  nnBench.train --config fold_${fold}.yaml --dataset Dataset001_Hippo
done

# Run inference
for fold in {0..4}; do
  nnBench.inference --config fold_${fold}.yaml --dataset Dataset001_Hippo
done
```

## Full Dataset Training (fold: -1)

Train on all data without validation split:

```bash
# Generate config
nnBench.plan --dataset Dataset001_Hippo --fold -1

# Train
nnBench.train --config fold_-1.yaml --dataset Dataset001_Hippo
```

## Key Differences

| Aspect | 5-Fold CV | Full Dataset (fold: -1) |
|--------|-----------|-------------------------|
| **When to Use** | Evaluation & benchmarking | Final production model |
| **Training Data** | 80% per fold | 100% of data |
| **Validation Split** | Yes (20% per fold) | No |
| **Training Time** | 5× single fold | Same as one fold |

## Tips

- **Parallel training**: Use different GPUs per fold with `CUDA_VISIBLE_DEVICES=<N>`
- **Resume training**: Use `--continue` flag if interrupted
- **Monitor progress**: `tail -f nnBench_results/Dataset001_Hippo/fold_0/training_history.json`
