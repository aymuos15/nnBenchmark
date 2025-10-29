# defining classes

**always include background (class 0) in `num_classes`**

## rule

`dataset.num_classes` = `model.out_channels` = `metrics.num_classes`

all must include background count

## example

background + 2 regions = 3 classes

```yaml
dataset:
  num_classes: 3

model:
  out_channels: 3

metrics:
  - type: DiceMetric
    include_background: false  # skip background in metrics
    num_classes: 3
```
