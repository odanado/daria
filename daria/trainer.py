import time

import torch


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
        self._reset_metrics()

        for batch in self.train_iter:
            loss, y_true, y_pred = self._one_iter(batch)
            self._update_metrics(loss, y_true, y_pred)

        self._store_metrics()

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

    def _reset_metrics(self):
        for m in self.metrics:
            m.reset()

    def _update_metrics(self, loss, y_true, y_pred):
        for m in self.metrics:
            m.update(loss, y_true, y_pred)

    def _store_metrics(self):
        for m in self.metrics:
            if m.name not in self.history:
                self.history[m.name] = []

            self.history[m.name].append(m.score())
