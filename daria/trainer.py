import time


from .metrics import _reset_metrics, _update_metrics, _store_metrics


class Trainer(object):
    def __init__(self, updater, plugins=[], metrics=[]):
        self.updater = updater
        self.plugins = plugins
        self.metrics = metrics

        self.history = {
            'time': [],
            'epoch': []
        }
        self.init_epoch = 0
        self.epoch = self.iteration = 0

        self.updater.train_iter.repeat = False

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
        if hasattr(self.updater.train_iter, 'init_epoch'):
            self.updater.train_iter.init_epoch()

        self.updater.model.train()
        _reset_metrics(self.metrics)

        for batch in self.updater.train_iter:
            loss, y_true, y_pred = self.updater.update_once(batch)
            _update_metrics(self.metrics, loss, y_true, y_pred)

        _store_metrics(self.metrics, self.history)

    def state_dict(self):
        checkpoint = {}
        checkpoint['model'] = self.updater.model.state_dict()
        checkpoint['optimizer'] = self.updater.optimizer.state_dict()
        checkpoint['history'] = self.history
        checkpoint['start_time'] = time.time() - self._start_time
        checkpoint['epoch'] = self.epoch
        checkpoint['iteration'] = self.iteration

        return checkpoint

    def load_state_dict(self, checkpoint):
        self.updater.model.load_state_dict(checkpoint['model'])
        self.updater.optimizer.load_state_dict(checkpoint['optimizer'])

        self.history = checkpoint['history']
        self._start_time = time.time() - checkpoint['start_time']
        self.init_epoch = self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
