import torch
from torch.autograd import Variable


class Updater(object):
    def __init__(self, model, optimizer, criterion, train_iter, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_iter = train_iter

        self.device = device

    def _is_cuda(self):
        return self.device is None or self.device >= 0

    def _to_numpy(self, x):
        if self._is_cuda():
            x = x.cpu()
        if isinstance(x, Variable):
            return x.data.numpy()
        if torch.is_tensor(x):
            return x.numpy()

        raise TypeError(
            'x should be Tensor or Variable. get {}'.format(type(x)))

    def update(self):
        raise NotImplementedError

    def _update_one(self, data, target):
        self.optimizer.zero_grad()
        answer = self.model(data)

        loss = self.criterion(answer, target)
        loss.backward()
        self.optimizer.step()

        return loss, answer


class TupleUpdater(Updater):
    def update(self):
        self.model.train()

        for data, target in self.train_iter:
            if self._is_cuda():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            loss, answer = self._update_one(data, target)

            y_true = self._to_numpy(target)
            y_pred = self._to_numpy(torch.max(answer, dim=1)[1])
            # self._update_metrics(loss.data[0], y_true, y_pred)
