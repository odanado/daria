import unittest
from mock import MagicMock

from daria import plugins
import io


class TestPrintReport(unittest.TestCase):
    def setUp(self):
        self.out = io.StringIO()
        self.print_report = plugins.PrintReport(
            ['time', 'epoch'], out=self.out)

    def test_call(self):
        trainer_mock = MagicMock()
        trainer_mock.history = {
            'time': [0],
            'epoch': [0]
        }

        self.print_report(trainer_mock)
        lines = self.out.getvalue().split('\n')
        self.assertEqual(list(map(int, lines[1].split())), [0, 0])


if __name__ == '__main__':
    unittest.main()
