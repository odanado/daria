import unittest
import mock
from daria import Updater


class TestUpdater(unittest.TestCase):
    def setUp(self):
        self.model_mock = mock.MagicMock()
        self.optimizer_mock = mock.MagicMock()
        self.criterion_mock = mock.MagicMock()
        self.train_iter_mock = mock.MagicMock()
        self.converter_mock = mock.MagicMock()
        self.updater = Updater(self.model_mock, self.optimizer_mock,
                               self.criterion_mock, self.train_iter_mock,
                               self.converter_mock)

    @mock.patch('torch.max', return_value=(None, 30))
    def test_update_once(self, max_mock):
        updater = self.updater
        updater.model.return_value = 100
        updater.converter = mock.MagicMock(return_value=(10, 20))
        loss_mock = mock.MagicMock()
        updater.calc_criterion = mock.MagicMock(return_value=loss_mock)

        loss, y_true, y_pred = updater.update_once((1, 2))

        updater.converter.assert_called_once_with((1, 2), updater.device)
        updater.optimizer.zero_grad.assert_called_once()
        updater.model.assert_called_once_with(10)

        updater.calc_criterion.assert_called_once_with(100, 20, (1, 2))
        loss_mock.backward.assert_called_once()
        updater.optimizer.step.assert_called_once()

        self.assertEqual(loss, loss_mock)

    def test_calc_criterion(self):
        updater = self.updater
        updater.calc_criterion(10, 20, 30)

        updater.criterion.assert_called_once_with(10, 20)
