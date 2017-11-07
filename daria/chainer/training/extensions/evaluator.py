import copy

from chainer.training import extensions
from chainer import reporter as reporter_module
from chainer.dataset import iterator as iterator_module

from torch import nn


class Evaluator(extensions.Evaluator):

    def __init__(self, iterator, target, loss_func, converter,
                 device=None, eval_hook=None):

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, nn.Module):
            target = {'main': target}
        self._targets = target
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.loss_func = loss_func

    def evaluate(self):
        iterator = self._iterators['main']
        loss_func = self.loss_func
        target = self._targets['main']
        target.eval()

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays, y = self.converter(batch, self.device)
                y_pred = target(*in_arrays)
                loss = loss_func(y_pred, y)
                reporter_module.report({'loss': loss.data[0]}, target)
                accuracy = y.data.eq(y_pred.max(dim=1)[1].data).sum() / len(y)
                reporter_module.report({'accuracy': accuracy}, target)

            summary.add(observation)

        return summary.compute_mean()
