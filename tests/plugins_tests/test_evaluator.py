import unittest
import mock

from daria.plugins import Evaluate
from daria.metrics import Accuracy, Loss


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.dev_iter_mock = mock.MagicMock()
        self.trainer_mock = mock.MagicMock()
        self.trainer_mock.history = {}
        self.model_mock = mock.MagicMock()
        self.criterion_mock = mock.MagicMock()
        self.converter_mock = mock.MagicMock()

        self.metrics_mock = [Accuracy(), Loss()]
        for m in self.metrics_mock:
            m.score = mock.MagicMock(return_value=10)
            m.update = mock.MagicMock()
            m.reset = mock.MagicMock()

        self.evaluator = Evaluate(self.model_mock, self.criterion_mock,
                                  self.dev_iter_mock, self.converter_mock,
                                  metrics=self.metrics_mock)

    def test_call_with_trainer(self):
        evaluator = self.evaluator
        evaluator.dev_iter.__iter__.return_value = [(2, 3), (12, 23)]
        self.evaluator._one_iter = mock.MagicMock(return_value=(0, 1, 2))

        evaluator(self.trainer_mock)

        evaluator._one_iter.assert_any_call((2, 3))
        evaluator._one_iter.assert_any_call((12, 23))

        self.model_mock.eval.assert_called_once()
        self.assertEqual(evaluator.history['dev/accuracy'][0], 10)
        self.assertEqual(evaluator.history['dev/loss'][0], 10)

        evaluator(self.trainer_mock)

        self.assertEqual(evaluator.history['dev/accuracy'][1], 10)
        self.assertEqual(evaluator.history['dev/loss'][1], 10)

    def test_call_without_trainer(self):
        evaluator = self.evaluator

        evaluator()

        self.model_mock.eval.assert_called_once()
        self.assertEqual(evaluator.history['dev/accuracy'][0], 10)
        self.assertEqual(evaluator.history['dev/loss'][0], 10)

        evaluator()
        self.assertEqual(len(evaluator.history['dev/accuracy']), 1)
        self.assertEqual(len(evaluator.history['dev/loss']), 1)

    @mock.patch('torch.max', return_value=(None, 30))
    def test_one_iter(self, max_mock):
        evaluator = self.evaluator
        evaluator.model.return_value = 2
        evaluator.converter.return_value = (1, 20)
        evaluator.criterion.return_value = 10

        loss, y_true, y_pred = evaluator._one_iter(None)

        evaluator.converter.assert_called_once_with(
            None, evaluator.device, train=False)
        evaluator.model.assert_called_once_with(1)
        evaluator.criterion.assert_called_once_with(2, 20)

        self.assertEqual(loss, 10)
        self.assertEqual(y_true, 20)
        self.assertEqual(y_pred, 30)
