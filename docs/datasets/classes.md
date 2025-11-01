# Defining Classes and Channels

When configuring your dataset and model, the number of **output classes** must be consistent across all components. This is separate from the number of **input channels** (modalities).

## Classes vs. Channels

- **Input channels**: Number of modalities/sequences (e.g., T1, T2, FLAIR = 3 channels)
- **Output classes**: Number of segmentation targets including background (e.g., background + tumor = 2 classes)

## The Rule for Classes

```
dataset.num_classes = model.out_channels = metrics.num_classes
```

**All must include the background class count (never count channels here).**

## Example

If your segmentation has a background + 2 regions, that's **3 classes total**.

**dataset config:**
```yaml
dataset:
  num_classes: 3
```

**model config:**
```yaml
model:
  out_channels: 3
```

**metrics config:**
```yaml
metrics:
  - type: DiceMetric
    include_background: false  # skip background in metrics calculation
    num_classes: 3
```

## Key Points

- Background class is always 0 and must be counted in `num_classes`
- Use `include_background: false` in metrics if you want to evaluate only the foreground classes
- Mismatch between these values will cause errors during training
- **Important**: `num_classes` refers to segmentation output classes, NOT input channels
  - A dataset with 4 input channels (T1, T2, FLAIR, T1gd) and 2 output classes (background + tumor) has `num_classes: 2`
- Input channels are defined separately in `channel_names` in dataset.json
