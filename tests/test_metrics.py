import unittest
import mock

import torch
from torch.autograd import Variable

from daria import metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics_mock = [mock.MagicMock() for i in range(2)]
        for i, m in enumerate(self.metrics_mock):
            type(m).name = mock.PropertyMock(return_value='mock{}'.format(i))
            type(m).score = mock.MagicMock(return_value=i * 10)

    def test_reset_metrics(self):
        metrics._reset_metrics(self.metrics_mock)

        self.metrics_mock[0].reset.assert_called_once()
        self.metrics_mock[1].reset.assert_called_once()

    def test_update_metrics(self):
        metrics._update_metrics(self.metrics_mock, 0, 1, 2)

        self.metrics_mock[0].update.assert_called_once_with(0, 1, 2)
        self.metrics_mock[1].update.assert_called_once_with(0, 1, 2)

    def test_store_metrics(self):
        history = {}

        metrics._store_metrics(self.metrics_mock, history)

        self.metrics_mock[0].score.assert_called_once()
        self.metrics_mock[1].score.assert_called_once()

        self.assertEqual(history['mock0'][0], 0)
        self.assertEqual(history['mock1'][0], 10)


class TestAccuracy(unittest.TestCase):
    def setUp(self):
        self.accuracy = metrics.Accuracy()

    def test(self):
        y_true = Variable(torch.zeros(1000))
        y_pred = Variable(torch.zeros(1000))

        self.accuracy.update(None, y_true, y_pred)
        y_pred = Variable(torch.ones(1000))
        self.accuracy.update(None, y_true, y_pred)

        self.assertEqual(self.accuracy.score(), 50)

    def test_reset(self):
        y_true = Variable(torch.zeros(1000))
        y_pred = Variable(torch.zeros(1000))
        self.accuracy.update(None, y_true, y_pred)
        self.accuracy.reset()

        self.assertEqual(self.accuracy.n_correct, 0)
        self.assertEqual(self.accuracy.n_total, 0)


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.loss = metrics.Loss()

    def test(self):
        self.loss.update(Variable(torch.FloatTensor([10])), None, None)
        self.loss.update(Variable(torch.FloatTensor([20])), None, None)

        self.assertEqual(self.loss.score(), 15)

    def test_reset(self):
        self.loss.update(Variable(torch.FloatTensor([10])), None, None)
        self.loss.reset()
        self.assertEqual(self.loss.sum_loss, 0)
        self.assertEqual(self.loss.n_total, 0)
