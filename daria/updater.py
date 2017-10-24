import torch


class Updater(object):
    def __init__(self, model, optimizer, criterion, train_iter,
                 converter, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_iter = train_iter
        self.converter = converter
        self.device = device

    def calc_criterion(self, answer, target, batch):
        return self.criterion(answer, target)

    def update_once(self, batch):
        data, target = self.converter(batch, self.device)
        self.optimizer.zero_grad()
        answer = self.model(data)

        loss = self.calc_criterion(answer, target, batch)
        loss.backward()
        self.optimizer.step()

        y_true = target
        y_pred = torch.max(answer, dim=1)[1]

        return loss, y_true, y_pred
