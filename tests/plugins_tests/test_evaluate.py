import unittest
import mock

from daria.plugins import Evaluate
from daria.metrics import Accuracy, Loss


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.dev_iter_mock = mock.MagicMock()
        self.trainer_mock = mock.MagicMock()
        self.trainer_mock.history = {}

        self.metrics_mock = [Accuracy(), Loss()]
        for m in self.metrics_mock:
            m.score = mock.MagicMock(return_value=10)
            m.update = mock.MagicMock()
            m.reset = mock.MagicMock()
        self.trainer_mock.metrics = self.metrics_mock

        self.evaluate = Evaluate(self.dev_iter_mock)

    def test_prepare(self):
        evaluate = self.evaluate
        evaluate.prepare(self.trainer_mock)

        self.assertIsNotNone(evaluate.model)
        self.assertIsNotNone(evaluate.criterion)
        self.assertIsNotNone(evaluate.converter)

        self.assertEqual(evaluate.metrics[0].name, 'dev/accuracy')
        self.assertEqual(evaluate.metrics[1].name, 'dev/loss')

    def test_call_with_trainer(self):
        evaluate = self.evaluate
        evaluate.dev_iter.__iter__.return_value = [(2, 3), (12, 23)]
        self.evaluate._one_iter = mock.MagicMock(return_value=(0, 1, 2))
        evaluate.prepare(self.trainer_mock)

        evaluate(self.trainer_mock)

        evaluate._one_iter.assert_any_call((2, 3))
        evaluate._one_iter.assert_any_call((12, 23))

        self.trainer_mock.model.eval.assert_called_once()
        self.assertEqual(evaluate.history['dev/accuracy'][0], 10)
        self.assertEqual(evaluate.history['dev/loss'][0], 10)

        evaluate(self.trainer_mock)

        self.assertEqual(evaluate.history['dev/accuracy'][1], 10)
        self.assertEqual(evaluate.history['dev/loss'][1], 10)

    def test_call_without_trainer(self):
        evaluate = self.evaluate
        evaluate.prepare(self.trainer_mock)

        evaluate()

        self.trainer_mock.model.eval.assert_called_once()
        self.assertEqual(evaluate.history['dev/accuracy'][0], 10)
        self.assertEqual(evaluate.history['dev/loss'][0], 10)

        evaluate()
        self.assertEqual(len(evaluate.history['dev/accuracy']), 1)
        self.assertEqual(len(evaluate.history['dev/loss']), 1)

    @mock.patch('torch.max', return_value=(None, 30))
    def test_one_iter(self, max_mock):
        evaluate = self.evaluate
        evaluate.prepare(self.trainer_mock)
        evaluate.model.return_value = 2
        evaluate.converter.return_value = (1, 20)
        evaluate.criterion.return_value = 10

        loss, y_true, y_pred = evaluate._one_iter(None)

        evaluate.converter.assert_called_once_with(
            None, evaluate.device, train=False)
        evaluate.model.assert_called_once_with(1)
        evaluate.criterion.assert_called_once_with(2, 20)

        self.assertEqual(loss, 10)
        self.assertEqual(y_true, 20)
        self.assertEqual(y_pred, 30)
