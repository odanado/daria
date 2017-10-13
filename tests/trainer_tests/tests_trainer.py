import time
import unittest
import mock

from daria import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.model_mock = mock.MagicMock()
        self.optimizer_mock = mock.MagicMock()
        self.criterion_mock = mock.MagicMock()
        self.train_iter_mock = mock.MagicMock()
        self.converter_mock = mock.MagicMock()

        self.metrics_mock = [mock.MagicMock() for i in range(2)]
        for i, m in enumerate(self.metrics_mock):
            type(m).name = mock.PropertyMock(return_value='mock{}'.format(i))
            type(m).score = mock.MagicMock(return_value=i * 10)

        self.trainer = Trainer(self.model_mock, self.optimizer_mock,
                               self.criterion_mock, self.train_iter_mock,
                               self.converter_mock, metrics=self.metrics_mock)

    def test_run(self):
        trainer = self.trainer
        trainer._one_step = mock.MagicMock()

        plugin_mock = mock.MagicMock()
        trainer.plugins = [plugin_mock]

        time.sleep(0.01)
        trainer.run(1)

        trainer._one_step.assert_called_once()
        plugin_mock.assert_called_once_with(trainer)

        self.assertEqual(trainer.history['epoch'][0], 0)
        self.assertGreater(trainer.history['time'][0], 0)

    def test_one_step(self):
        trainer = self.trainer
        trainer._one_iter = mock.MagicMock(return_value=(0, 1, 2))
        trainer._reset_metrics = mock.MagicMock()
        trainer._update_metrics = mock.MagicMock()
        trainer._store_metrics = mock.MagicMock()

        trainer.train_iter.__iter__.return_value = [(1, 2), (10, 20)]

        trainer._one_step()

        trainer.model.train.assert_called_once()
        trainer._reset_metrics.assert_called()
        trainer._one_iter.assert_any_call((1, 2))
        trainer._one_iter.assert_any_call((10, 20))
        trainer._update_metrics.assert_called_with(0, 1, 2)
        trainer._store_metrics.assert_called_once()

    def test_one_iter(self):
        trainer = self.trainer
        trainer.converter = mock.MagicMock(return_value=(10, 20))
        import torch
        answer = torch.rand(5, 5)
        trainer.model.return_value = answer
        loss_mock = mock.MagicMock()
        trainer.criterion.return_value = loss_mock

        loss, y_true, y_pred = trainer._one_iter((1, 2))

        trainer.converter.assert_called_once_with((1, 2), trainer.device)
        trainer.optimizer.zero_grad.assert_called_once()
        trainer.model.assert_called_once_with(10)

        trainer.criterion.assert_called_once_with(answer, 20)
        loss_mock.backward.assert_called_once()
        trainer.optimizer.step.assert_called_once()

        self.assertEqual(loss, loss_mock)

    def test_reset_metrics(self):
        trainer = self.trainer

        trainer._reset_metrics()

        for m in self.metrics_mock:
            m.reset.assert_called_once()

    def test_update_metrics(self):
        trainer = self.trainer

        trainer._update_metrics(1, 2, 3)

        for m in self.metrics_mock:
            m.update.assert_called_once_with(1, 2, 3)

    def test_store_metrics(self):
        trainer = self.trainer

        trainer._store_metrics()

        for m in self.metrics_mock:
            m.score.assert_called_once()
        self.assertEqual(trainer.history['mock0'][0], 0)
        self.assertEqual(trainer.history['mock1'][0], 10)
