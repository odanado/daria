import time

import torch

from .metrics import _reset_metrics, _update_metrics, _store_metrics


class Trainer(object):
    def __init__(self, model, optimizer, criterion, train_iter,
                 converter, device=None, plugins=[], metrics=[]):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_iter = train_iter
        self.converter = converter
        self.device = device
        self.plugins = plugins
        self.metrics = metrics

        self.history = {
            'time': [],
            'epoch': []
        }
        self.init_epoch = 0

        self.train_iter.repeat = False

        self._start_time = None

    def run(self, epochs):
        for plugin in self.plugins:
            plugin.prepare(self)

        if self._start_time is None:
            self._start_time = time.time()

        for epoch in range(self.init_epoch, epochs):
            self.history['epoch'].append(epoch)
            self._one_step()
            self.history['time'].append(time.time() - self._start_time)

            for plugin in self.plugins:
                plugin(self)

    def _one_step(self):
        if hasattr(self.train_iter, 'init_epoch'):
            self.train_iter.init_epoch()

        self.model.train()
        _reset_metrics(self.metrics)

        for batch in self.train_iter:
            loss, y_true, y_pred = self._one_iter(batch)
            _update_metrics(self.metrics, loss, y_true, y_pred)

        _store_metrics(self.metrics, self.history)

    def _one_iter(self, batch):
        data, target = self.converter(batch, self.device)
        self.optimizer.zero_grad()
        answer = self.model(data)

        loss = self.criterion(answer, target)
        loss.backward()
        self.optimizer.step()

        y_true = target
        y_pred = torch.max(answer, dim=1)[1]

        return loss, y_true, y_pred

    def state_dict(self):
        checkpoint = {}
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['history'] = self.history
        checkpoint['start_time'] = time.time() - self._start_time

        return checkpoint

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        self._start_time = time.time() - checkpoint['start_time']
