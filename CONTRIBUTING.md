# Contributing to nnBenchmark

We welcome contributions to nnBenchmark! This guide will help you get started.

## Documentation

We encourage documentation contributions, especially those that:

- **Clarify terminology** - Help users understand key concepts in medical imaging and segmentation
- **Improve existing docs** - Fix unclear explanations, add examples, or update outdated information
- **Add new guides** - Document new features, workflows, or best practices
- **Update terminology** - Ensure consistent use of terms across the codebase and documentation

### Before Making Documentation Changes

**Please raise an issue first** to discuss your proposed changes. This helps ensure:
- Changes align with project goals and documentation structure
- Effort isn't duplicated if someone else is working on the same area
- We maintain consistency in terminology and style

### Terminology Guidelines

nnBenchmark uses specific terminology to maintain consistency:

- **Channel** - A single imaging input (imaging technique like MRI/CT, or a specific sequence like T1/T2/FLAIR)
- **Case** - A single patient/subject's complete dataset with all channels and labels
- **Class** - A segmentation target (background is always class 0)

For more details, see [docs/terminology.md](docs/terminology.md).

---

*More contribution guidelines coming soon.*
