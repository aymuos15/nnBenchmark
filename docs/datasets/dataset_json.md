# dataset.json

The `dataset.json` file contains metadata about your dataset. It must be placed in the root of your dataset folder.

## Schema

```json
{
  "name": "Dataset Name",
  "description": "Brief description of the dataset",
  "labels": {
    "background": 0,
    "class_name_1": 1,
    "class_name_2": 2
  },
  "channel_names": {
    "0": "T2",
    "1": "ADC"
  },
  "numTraining": 260,
  "file_ending": ".nii.gz"
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Short dataset name (e.g., "Hippo") |
| `description` | string | Human-readable description of the dataset |
| `labels` | object | Maps class names to integer IDs (0 = background) |
| `channel_names` | object | Maps channel indices to channel names (e.g., "0": "T2", "1": "ADC") |
| `numTraining` | integer | Number of training cases |
| `file_ending` | string | File extension for all images/labels (e.g., ".nii.gz") |

## Example: Single Channel

```json
{
  "name": "Hippo",
  "description": "hippocampus segmentation",
  "labels": {
    "background": 0,
    "Anterior": 1,
    "Posterior": 2
  },
  "channel_names": {
    "0": "MRI"
  },
  "numTraining": 260,
  "file_ending": ".nii.gz"
}
```

## Example: Multi-Channel (e.g., Brain Tumor)

```json
{
  "name": "BrainTumour",
  "description": "brain tumor segmentation with 4 MRI sequences",
  "labels": {
    "background": 0,
    "tumor": 1
  },
  "channel_names": {
    "0": "FLAIR",
    "1": "T1w",
    "2": "T1gd",
    "3": "T2w"
  },
  "numTraining": 369,
  "file_ending": ".nii.gz"
}
```

## Notes

- Always include the background class (0)
- All class IDs must be sequential starting from 0
- The number of labels should match `num_classes` in your config (including background)
- **Multi-channel**: Each channel is stored as a separate file (e.g., `case_001_0000.nii.gz`, `case_001_0001.nii.gz`)
- **Channel consistency**: All images must have the same channels in the same order
- **Channel order matters**: The channel index suffix (0000, 0001, etc.) must match the order in `channel_names`
