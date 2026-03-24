"""Learning Rate Scheduler matching nnU-Net's PolyLRScheduler exactly.

Reference: nnunetv2/training/lr_scheduler/polylr.py
"""

from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    """Polynomial LR decay matching nnU-Net's implementation.

    lr = initial_lr * (1 - epoch / max_steps) ^ exponent

    Uses the same counter pattern as nnU-Net: auto-increments when
    called without an explicit epoch, guards against the implicit
    step(-1) from _LRScheduler.__init__.
    """

    def __init__(self, optimizer, initial_lr: float, max_epochs: int, exponent: float = 0.9, **kwargs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_epochs
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, last_epoch=-1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr
