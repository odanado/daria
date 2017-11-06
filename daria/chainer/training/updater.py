from __future__ import division


from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.training import StandardUpdater as BaseStandardUpdater
from chainer import reporter


class StandardUpdater(BaseStandardUpdater):

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays, y = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func

        optimizer.zero_grad()

        y_pred = optimizer.target(*in_arrays)
        loss = loss_func(y_pred, y)

        loss.backward()
        optimizer.step()

        reporter.report({'loss': loss.data[0]}, optimizer.target)

        accuracy = y.data.eq(y_pred.max(dim=1)[1].data).sum() / len(y)
        reporter.report({'accuracy': accuracy}, optimizer.target)
