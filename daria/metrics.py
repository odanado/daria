class AbstractMetrics(object):
    def __init__(self):
        self.reset()

    def update(self, loss, y_true, y_pred):
        raise NotImplementedError()

    def score(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class Accuracy(AbstractMetrics):
    name = 'accuracy'

    def update(self, loss, y_true, y_pred):
        self.n_correct += (y_true == y_pred).sum().data[0]
        self.n_total += len(y_true)

    def score(self):
        return 100 * self.n_correct / self.n_total

    def reset(self):
        self.n_correct = 0
        self.n_total = 0


class Loss(AbstractMetrics):
    name = 'loss'

    def update(self, loss, y_true, y_pred):
        self.sum_loss += loss.data[0]
        self.n_total += 1

    def score(self):
        return self.sum_loss / self.n_total

    def reset(self):
        self.sum_loss = 0
        self.n_total = 0
