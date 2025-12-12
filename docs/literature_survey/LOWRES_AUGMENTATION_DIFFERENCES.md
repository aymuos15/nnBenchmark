# EXACT DIFFERENCES: RandZoomd vs SimulateLowResolutionTransform

**Date**: December 12, 2025  
**Status**: Detailed Analysis Complete  
**Recommendation**: Option B or C for exact nnU-Net match

---

## Parameter Comparison

### nnU-Net (batchgenerators.SimulateLowResolutionTransform)
```python
SimulateLowResolutionTransform(
    zoom_range=(0.5, 1),              # nnUNetTrainer.py:736
    per_channel=True,                  # Line 736
    p_per_channel=0.5,                 # Line 737
    order_downsample=0,                # Line 738 - NEAREST NEIGHBOR for downsampling
    order_upsample=3,                  # Line 738 - CUBIC for upsampling
    p_per_sample=0.25,                 # Line 738 - Probability of applying transform
    ignore_axes=ignore_axes            # Line 739
)
```

Source: [nnUNetTrainer.py:736-739](https://github.com/MIC-DKFZ/nnUNet/blob/v2.4.1/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L736-L739)

### nnBenchmark (MONAI RandZoomd)
```yaml
- type: RandZoomd
  keys: [image, label]
  prob: 0.25                           # Same as p_per_sample
  min_zoom: 0.5
  max_zoom: 1.0
  mode: [bilinear, nearest]            # bilinear for images, nearest for labels
  padding_mode: edge                   # Edge padding
```

Source: `src/planning/yaml_generator.py:537-543`

---

## EXACT DIFFERENCES BREAKDOWN

### 1. **Interpolation Order (MAJOR DIFFERENCE)** 🔴

| Aspect | nnU-Net | nnBenchmark | Consequence |
|--------|---------|------------|-------------|
| **Downsample Interpolation** | order=0 (Nearest Neighbor) | bilinear | Sharp vs Smooth |
| **Upsample Interpolation** | order=3 (Cubic) | bilinear | Smooth vs Smooth |
| **Combined Effect** | NN↓ → Cubic↑ | Bilinear throughout | Different artifacts |

**nnU-Net's approach**:
- Uses nearest-neighbor (order=0) when downsampling
  - Creates blocky, pixelated patterns
  - Loses high-frequency information sharply
- Uses cubic (order=3) when upsampling
  - Smooths the sharp artifacts
  - Creates anti-aliasing effect
- **Result**: Realistic low-resolution imaging artifacts

**nnBenchmark's approach**:
- Uses bilinear (order=1) directly
- Smoothly reduces resolution without sharp artifacts
- **Result**: Smooth but potentially unrealistic artifacts

**Training Impact**: 
- Models trained with nnU-Net's approach learn to handle realistic low-resolution images
- Models trained with nnBenchmark's approach may not see true alias artifacts

---

### 2. **Resampling Algorithm (MAJOR DIFFERENCE)** 🔴

| Aspect | nnU-Net | nnBenchmark |
|--------|---------|------------|
| **Strategy** | Two-phase (downsample→upsample) | Single-phase (zoom) |
| **Library** | scipy.ndimage.zoom | torch.nn.functional.interpolate |
| **Steps** | 1. Downsample to target_shape, 2. Upsample back to original | Direct scaling to zoom factor |

**nnU-Net's Implementation** (resample_augmentations.py:89-92):
```python
downsampled = resize(data_sample[c].astype(float), target_shape, 
                     order=0, mode='edge', anti_aliasing=False)
data_sample[c] = resize(downsampled, shp, 
                       order=3, mode='edge', anti_aliasing=False)
```

- Two separate scipy.ndimage.zoom calls
- First with nearest-neighbor (order=0)
- Second with cubic (order=3)
- Explicit `anti_aliasing=False` on both

**nnBenchmark's Implementation** (MONAI RandZoomd):
```python
# Internal PyTorch interpolation
torch.nn.functional.interpolate(input, scale_factor=zoom_factor, 
                               mode='bilinear', align_corners=False)
```

- Single PyTorch interpolation call
- No explicit two-phase approach
- Bilinear interpolation throughout

**Why This Matters**:
The two-phase approach creates a fundamentally different signal:
1. Downsampling with NN → Information loss + aliasing
2. Upsampling with Cubic → Smooth recovery but aliasing preserved

This mimics real low-resolution imaging (sensor degradation).

---

### 3. **Per-Channel Behavior (MODERATE DIFFERENCE)** 🟡

| Aspect | nnU-Net | nnBenchmark |
|--------|---------|------------|
| **per_channel** | True | Not configurable |
| **p_per_channel** | 0.5 | N/A |
| **Behavior** | 50% chance each channel gets different augmentation | All channels augmented identically |
| **Zoom Factor Variety** | Each channel gets independent zoom factor | Single zoom factor for all channels |

**nnU-Net's per_channel logic** (resample_augmentations.py:75-92):
```python
for c in channels:
    if np.random.uniform() < p_per_channel:  # 50% probability
        # Each channel can decide independently
        if per_channel:
            zoom = uniform(zoom_range[0], zoom_range[1])  # New random zoom per channel
            # Apply with potentially different factor
```

**nnBenchmark's behavior**:
- RandZoomd applies same zoom to all channels
- No per-channel probability control in YAML
- All channels see identical transformation

**Training Impact**:
- nnU-Net: More channel diversity, models learn robustness to selective channel degradation
- nnBenchmark: Less diversity, all channels degrade identically

---

### 4. **Anti-aliasing Filter (MINOR DIFFERENCE)** 🟡

| Aspect | nnU-Net | nnBenchmark |
|--------|---------|------------|
| **anti_aliasing** | Explicitly False | Implicit (PyTorch default) |
| **Effect** | No smoothing filter before downsampling | May have implicit filtering |

**nnU-Net's explicit setting** (resample_augmentations.py:90-91):
```python
resize(data_sample[c], target_shape, order=0, mode='edge', anti_aliasing=False)
```

- No low-pass filter applied
- Results in harsh aliasing artifacts

**nnBenchmark's behavior**:
- MONAI uses torch.nn.functional.interpolate
- PyTorch's bilinear mode has implicit smoothing
- May reduce aliasing artifacts

**Impact on Training**:
- Minor frequency-domain differences
- nnU-Net has more high-frequency aliasing
- Could affect edge detection performance slightly

---

### 5. **Ignore Axes Feature (MODERATE DIFFERENCE)** 🟡

| Aspect | nnU-Net | nnBenchmark |
|--------|---------|------------|
| **ignore_axes support** | Yes | No |
| **Use case** | Preserve Z-axis for anisotropic data | Always zoom all axes |
| **Example** | ignore_axes=[2] keeps Z resolution | No equivalent |

**nnU-Net's ignore_axes logic** (resample_augmentations.py:68-70, 85-87):
```python
if ignore_axes is not None:
    for i in ignore_axes:
        target_shape[i] = shp[i]  # Don't change this axis
```

Allows preserving specific axes during downsampling.

**nnBenchmark's limitation**:
- RandZoomd zooms all spatial dimensions uniformly
- For anisotropic data (e.g., CT with 5mm slice spacing), cannot preserve Z-axis resolution
- This is a **feature gap**

**Impact**:
- For isotropic data: Negligible
- For anisotropic data: Misses nnU-Net's fine-tuning capability

---

### 6. **Edge/Padding Handling (MINOR DIFFERENCE)** 🟢

| Aspect | nnU-Net | nnBenchmark |
|--------|---------|------------|
| **Edge mode** | 'edge' in scipy zoom | 'edge' in padding |
| **Application** | During downsample AND upsample | After zoom operation |

Both use edge mode (reflect boundary values), so impact is minimal.

---

## Summary Table

| Aspect | nnU-Net | nnBenchmark | Severity | Impact |
|--------|---------|------------|----------|--------|
| **Interpolation Order** | order_down=0, order_up=3 | bilinear | 🔴 **MAJOR** | Different artifact patterns |
| **Algorithm** | Two-phase (down↓→up↑) | Single-phase | 🔴 **MAJOR** | Fundamentally different signal |
| **Per-Channel** | per_channel=True, p=0.5 | No control | 🟡 **MODERATE** | Less channel diversity |
| **Anti-aliasing** | Disabled (False) | Implicit | 🟡 **MINOR** | High-freq difference |
| **Ignore Axes** | Supported | Not supported | 🟡 **MODERATE** | Can't optimize for anisotropic |
| **Edge Mode** | edge | edge | 🟢 **MINOR** | Nearly identical |
| **Zoom Range** | (0.5, 1) | (0.5, 1.0) | 🟢 **IDENTICAL** | None |
| **Probability** | p_per_sample=0.25 | prob=0.25 | 🟢 **IDENTICAL** | None |

---

## Functional Equivalence Assessment

### Current Documentation Status
**hyperparameters.md Line 77**: "✅ RandZoomd (MONAI equivalent) ... ✅ Functionally equivalent"

### Revised Assessment: **NOT FUNCTIONALLY EQUIVALENT** ❌

The implementations create **fundamentally different augmentation effects**:

**nnU-Net's Approach** creates realistic low-resolution artifacts:
1. Aggressively downsample with nearest-neighbor (NN at order=0)
   - Causes information loss
   - Creates blocky patterns
   - Preserves aliasing artifacts
   
2. Smooth upsampling with cubic (Cubic at order=3)
   - Recovers spatial smoothness
   - Preserves alias artifacts learned from step 1
   
3. **Result**: Realistic low-resolution imaging artifacts that mimic sensor degradation

**nnBenchmark's Approach** creates smooth scaling artifacts:
1. Direct bilinear interpolation
   - Smooth information loss
   - No blocky patterns
   - Implicit anti-aliasing
   
2. **Result**: Smooth but potentially unrealistic artifacts

### Training Impact

Different augmentation signals → **potentially different model behavior**:

| Scenario | nnU-Net | nnBenchmark |
|----------|---------|------------|
| **Low-res test image** | Model expects aliasing artifacts | Model expects smooth degradation |
| **High-frequency details** | Learned under alias conditions | Learned under smooth conditions |
| **Edge robustness** | Better at handling blocky patterns | Better at handling smooth patterns |

---

## Recommendation for Action

### Option A: **Keep RandZoomd (Document Limitation)**
- **Effort**: 30 minutes (docs only)
- **Result**: Explicit acknowledgment that implementations differ
- **Trade-off**: Stays with MONAI, loses exact nnU-Net behavior
- **When to choose**: If prioritizing code simplicity and MONAI integration

### Option B: **Implement Custom SimulateLowResolutionTransform** ✅ RECOMMENDED
- **Effort**: 2-3 hours
- **Result**: Exact nnU-Net match, independent of external libraries
- **Trade-off**: More custom code, but guaranteed equivalence
- **How**:
  1. Create MONAI-compatible wrapper around scipy two-phase zoom
  2. Implement per-channel probability logic
  3. Support ignore_axes parameter
  4. Validate against nnU-Net behavior with test images
- **When to choose**: If prioritizing exact nnU-Net compatibility

### Option C: **Use nnU-Net's Transform Directly**
- **Effort**: 1-2 hours (import + wrapper)
- **Result**: 100% nnU-Net compatibility, guaranteed
- **Trade-off**: Adds nnU-Net as runtime dependency
- **How**:
  1. Import batchgenerators SimulateLowResolutionTransform
  2. Create MONAI-compatible adapter
  3. Integrate into YAML transform pipeline
- **When to choose**: If nnU-Net is acceptable dependency

---

## Next Steps

1. **Review this analysis** with team
2. **Choose implementation approach** (A, B, or C)
3. **Update hyperparameters.md** with corrected assessment
4. **Implement chosen solution** (if B or C)
5. **Add regression tests** comparing augmented outputs

---

**Document Version**: 1.0  
**Last Updated**: December 12, 2025  
**Status**: Ready for Decision
