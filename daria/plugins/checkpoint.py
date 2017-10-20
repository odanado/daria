import os
import torch
from ..plugin import Plugin


class Checkpoint(Plugin):
    name = 'checkpoint'

    def __init__(self, save_path, target_metrics, mode):
        self.save_path = save_path
        self.target_metrics = target_metrics
        self.best = None

        if mode == 'min':
            self.compare = lambda x, y: x < y
        elif mode == 'max':
            self.compare = lambda x, y: x > y
        else:
            raise ValueError('mode should be min or max')

    def __call__(self, trainer):
        checkpoint = trainer.state_dict()

        checkpoint_path = os.path.join(
            self.save_path, 'latest_checkpoint.pth.tar')
        torch.save(checkpoint, checkpoint_path)

        score = trainer.history[self.target_metrics][-1]
        if self.best is None or self.compare(score, self.best):
            checkpoint_path = os.path.join(
                self.save_path, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, checkpoint_path)
            self.best = score
