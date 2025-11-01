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
: A dataset where one axis has significantly different spacing (>3x) and fewer voxels (<25%) than other axes, requiring special handling in model architecture.

**Reference Label**
: The annotated segmentation provided by human experts, used as the reference standard for training and evaluating models. We use "reference label" instead of "ground truth" because human annotations in medical imaging are inherently subjective and suffer from inter- and intra-rater variability. These annotations represent expert consensus rather than absolute truth. This terminology acknowledges the limitations and subjective nature of manual annotations while recognizing their value as reference standards. See: [Kofler et al., "Approaching Peak Ground Truth"](https://arxiv.org/abs/2301.00243).

> **Note**: You may see "ground truth" in older medical imaging literature, but we intentionally avoid this term in nnBenchmark as it falsely implies that human annotations represent absolute, objective truth.

## Data Splits

**Train Set**
: The subset of data used to train the model. The model learns from this data by adjusting its parameters to minimize prediction errors.

**Validation Set (Val Set)**
: The subset of data used to evaluate the model during training and tune hyperparameters. Not used for training, but helps guide model development decisions.

**Test Set**
: The subset of data held out for final model evaluation. Aims to provides an unbiased assessment of model performance on unseen data.

## Processes

**Model**
: A deep learning architecture with learnable parameters (weights and biases) that is trained to perform medical image segmentation.

**Training**
: The process of teaching a model to perform segmentation by iteratively adjusting its parameters based on training data to minimize prediction errors.

**Inference**
: The process of using a trained model to make predictions.

**Prediction**
: The segmentation output produced by a trained model during inference. Predictions are compared against reference labels to evaluate model performance. Unlike reference labels (which are manual expert annotations), predictions are generated automatically by the model.

## Segmentation Classes

**Class**
: A category or label in segmentation. Classes represent different regions or structures in the image.

- **Background class** - Class ID 0, representing regions that are not part of the segmentation targets. Always present and counted in `num_classes`.
- **Foreground class** - Any segmentation target class. Example: if segmenting tumors, tumor is a foreground class.

*This terminology guide will be expanded as new concepts are introduced.*
