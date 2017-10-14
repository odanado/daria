import copy

import torch

from ..plugin import Plugin
from ..metrics import _reset_metrics, _update_metrics, _store_metrics


class Evaluate(Plugin):
    name = 'evaluate'
    prefix = 'dev'

    def __init__(self, dev_iter, model=None, criterion=None,
                 converter=None, device=None):
        self.dev_iter = dev_iter
        self.model = model
        self.criterion = criterion
        self.converter = converter
        self.device = device

    def prepare(self, trainer):
        if self.model is None:
            self.model = trainer.model

        if self.criterion is None:
            self.criterion = trainer.criterion

        if self.converter is None:
            self.converter = trainer.converter

        self.metrics = copy.deepcopy(trainer.metrics)
        for m in self.metrics:
            m.name = self.prefix + '/' + m.name

    def __call__(self, trainer=None):
        if hasattr(self.dev_iter, 'init_epoch'):
            self.dev_iter.init_epoch()

        if trainer is not None:
            self.history = trainer.history
        else:
            self.history = {}

        self.model.eval()
        _reset_metrics(self.metrics)

        for batch in self.dev_iter:
            loss, y_true, y_pred = self._one_iter(batch)
            _update_metrics(self.metrics, loss, y_true, y_pred)

        _store_metrics(self.metrics, self.history)

    def _one_iter(self, batch):
        data, target = self.converter(batch, self.device, train=False)
        answer = self.model(data)
        loss = self.criterion(answer, target)

        y_true = target
        y_pred = torch.max(answer, dim=1)[1]

        return loss, y_true, y_pred
