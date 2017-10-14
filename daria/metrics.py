def _reset_metrics(metrics):
    for m in metrics:
        m.reset()


def _update_metrics(metrics, loss, y_true, y_pred):
    for m in metrics:
        m.update(loss, y_true, y_pred)


def _store_metrics(metrics, history):
    for m in metrics:
        if m.name not in history:
            history[m.name] = []

        history[m.name].append(m.score())


class Metrics(object):
    def __init__(self):
        self.reset()

    def update(self, loss, y_true, y_pred):
        raise NotImplementedError()

    def score(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class Accuracy(Metrics):
    name = 'accuracy'

    def update(self, loss, y_true, y_pred):
        self.n_correct += (y_true.data == y_pred.data).sum()
        self.n_total += len(y_true)

    def score(self):
        return 100 * self.n_correct / self.n_total

    def reset(self):
        self.n_correct = 0
        self.n_total = 0


class Loss(Metrics):
    name = 'loss'

    def update(self, loss, y_true, y_pred):
        self.sum_loss += loss.data[0]
        self.n_total += 1

    def score(self):
        return self.sum_loss / self.n_total

    def reset(self):
        self.sum_loss = 0
        self.n_total = 0
