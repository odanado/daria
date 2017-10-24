import time
import unittest
import mock

from daria import Trainer


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.updater_mock = mock.MagicMock()

        self.metrics_mock = [mock.MagicMock() for i in range(2)]
        for i, m in enumerate(self.metrics_mock):
            type(m).name = mock.PropertyMock(return_value='mock{}'.format(i))
            type(m).score = mock.MagicMock(return_value=i * 10)

        self.trainer = Trainer(self.updater_mock, metrics=self.metrics_mock)

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
        trainer.updater.update_once = mock.MagicMock(return_value=(0, 1, 2))
        trainer._reset_metrics = mock.MagicMock()
        trainer._update_metrics = mock.MagicMock()
        trainer._store_metrics = mock.MagicMock()

        trainer.updater.train_iter.__iter__.return_value = [(1, 2), (10, 20)]

        trainer._one_step()

        trainer.updater.model.train.assert_called_once()
        trainer.updater.update_once.assert_any_call((1, 2))
        trainer.updater.update_once.assert_any_call((10, 20))

        for m in self.metrics_mock:
            m.reset.assert_called_once()

        for m in self.metrics_mock:
            m.update.assert_called_with(0, 1, 2)

        for m in self.metrics_mock:
            m.score.assert_called_once()

    def test_state_dict(self):
        self.updater_mock.model.state_dict.return_value = 10
        self.updater_mock.optimizer.state_dict.return_value = 20
        self.trainer._start_time = time.time()
        self.trainer.epoch = 10

        time.sleep(0.01)
        checkpoint = self.trainer.state_dict()
        time.sleep(0.1)

        self.trainer = Trainer(self.updater_mock, metrics=self.metrics_mock)
        self.trainer.load_state_dict(checkpoint)

        self.assertGreater(time.time() - self.trainer._start_time, 0.01)
        self.assertLess(time.time() - self.trainer._start_time, 0.1)
        self.assertEqual(self.trainer.init_epoch, 10)
