# Terminology

A reference guide for common terms used in nnBenchmark and medical image segmentation.

## Dataset

**Channel**
: A single imaging input. This can refer to either an imaging technique (e.g., MRI, CT, ultrasound) or a specific sequence (e.g., T1-weighted, T2-weighted, FLAIR, ADC). Examples: CT, T1-weighted MRI, T2-weighted MRI, FLAIR, ADC.

> **Note**: In nnBenchmark, we use "channel" for simplicity to refer to all imaging inputs—both the imaging technique and individual sequences within that technique. This unified terminology simplifies documentation and reduces confusion. In medical literature, these may be distinguished as "modality" vs "sequence," but here they are all channels.

**Multi-channel**
: A dataset where each case has multiple input channels. Example: Brain tumor data with T1, T2, FLAIR, and T1-gadolinium sequences = 4 channels.

**Case**
: A single patient or individual's complete imaging dataset with all channels and corresponding labels.

**Target Spacing**
: The voxel spacing used for planning calculations, typically median spacing with anisotropic adjustment for datasets with one axis significantly coarser than others.

**Anisotropic Dataset**
: A dataset where one axis has significantly different spacing (>3x) and fewer voxels (<25%) than other axes, requiring special handling in network architecture.

## Segmentation Classes

**Class**
: A category or label in segmentation. Classes represent different regions or structures in the image.

- **Background class** - Class ID 0, representing regions that are not part of the segmentation targets. Always present and counted in `num_classes`.
- **Foreground class** - Any segmentation target class. Example: if segmenting tumors, tumor is a foreground class.

*This terminology guide will be expanded as new concepts are introduced.*
