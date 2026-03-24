# LR Schedule Step Timing Mismatch with nnU-Net

## Problem

nnBenchmark steps the LR scheduler on `EPOCH_COMPLETED`, while nnU-Net steps it on epoch start (`on_train_epoch_start`). This creates an off-by-one: nnBenchmark trains each epoch with the previous epoch's LR.

With 5 epochs and PolyLR (exponent=0.9):

| Epoch | nnU-Net LR | nnBenchmark LR |
|-------|-----------|----------------|
| 0     | 0.010000  | 0.010000       |
| 1     | 0.008181  | 0.010000       |
| 2     | 0.006314  | 0.008181       |
| 3     | 0.004384  | 0.006314       |
| 4     | 0.002349  | 0.004384       |

nnBenchmark uses a consistently higher LR, causing overshooting in later epochs where fine convergence matters most.

## Fix

Changed `Events.EPOCH_COMPLETED` to `Events.EPOCH_STARTED` for the LrScheduleHandler in `src/engines/ignite_utils/trainer.py`.

## Reference

nnU-Net: `nnUNetTrainer.on_train_epoch_start()` calls `self.lr_scheduler.step(self.current_epoch)`.
