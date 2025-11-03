# Workflow: CLI Commands

## nnBench.plan

```bash
nnBench.plan --dataset <name>                      # Auto-generate optimal config
nnBench.plan --dataset <name> --gpu-memory-gb <gb> # Set target GPU memory
nnBench.plan --dataset <name> --fold <n>           # Specify fold number (default: 0)
nnBench.plan --dataset <name> --output <path>      # Custom output path for config
nnBench.plan --dataset <name> --verbose            # Enable detailed logging
nnBench.plan --dataset <name> --num-workers <n>    # Set parallel workers for fingerprinting
```

## nnBench.train

```bash
nnBench.train --config <config> --dataset <name>   # Train with config (dataset required for relative paths)
nnBench.train --config <config> --dataset <name> --continue  # Resume from last checkpoint
nnBench.train --config <config> --dataset <name> -c          # Resume (short flag)
```

## nnBench.inference

```bash
nnBench.inference --config <config> --dataset <name>              # Run inference on test set
nnBench.inference --config <config> --dataset <name> --model <path>  # Use specific model weights
nnBench.inference --config <config> --dataset <name> --use-val-split  # Use validation split instead of test set
```

## nnBench.plot

```bash
nnBench.plot --config <config> --dataset <name>    # Generate all plots from results
```

# End to End Example for the default config generated for a dataset.
```bash
nnBench.plan --dataset Dataset002_HippocampusMedDecathalon --verbose # Plan
nnBench.train --config fold_0.yaml --dataset Dataset001_Cellpose # Train
nnBench.inference --config fold_0.yaml --dataset Dataset001_Cellpose # Inference
nnBench.plot --config fold_0.yaml --dataset Dataset001_Cellpose # Plot
```
