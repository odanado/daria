import unittest
import mock

from daria.plugins import Checkpoint


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.checkpoint = Checkpoint('/tmp', 'loss', 'min')
        self.trainer_mock = mock.MagicMock()
        self.trainer_mock.history = {'loss': [0.5, 0.1], 'accuracy': [30, 50]}
        self.trainer_mock.state_dict.return_value = {}

    @mock.patch('torch.save')
    def test_call(self, save_mock):
        self.checkpoint(self.trainer_mock)
        save_mock.assert_any_call({}, '/tmp/latest_checkpoint.pth.tar')
        save_mock.assert_any_call({}, '/tmp/best_checkpoint.pth.tar')
        self.assertEqual(save_mock.call_count, 2)

        self.checkpoint(self.trainer_mock)
        self.assertEqual(save_mock.call_count, 3)

        self.trainer_mock.history['loss'].append(0.01)
        self.checkpoint(self.trainer_mock)
        self.assertEqual(save_mock.call_count, 5)

    @mock.patch('torch.save')
    def test_call_max(self, save_mock):
        self.checkpoint = Checkpoint('/tmp', 'accuracy', 'max')
        self.checkpoint(self.trainer_mock)
        self.assertEqual(save_mock.call_count, 2)

        self.checkpoint(self.trainer_mock)
        self.assertEqual(save_mock.call_count, 3)

        self.trainer_mock.history['accuracy'].append(100)
        self.checkpoint(self.trainer_mock)
        self.assertEqual(save_mock.call_count, 5)

    def test_init_raise_value_error(self):
        with self.assertRaises(ValueError):
            Checkpoint('/tmp', 'accuracy', 'poyo')
