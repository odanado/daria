import unittest
from mock import MagicMock


from daria.converters import tuple_converter


class TestTupleConverter(unittest.TestCase):

    def test_cpu(self):
        data = MagicMock()
        target = MagicMock()

        tuple_converter((data, target), -1)

        data.cuda.assert_not_called()
        target.cuda.assert_not_called()

    def test_cuda(self):
        data = MagicMock()
        target = MagicMock()

        tuple_converter((data, target), 0)

        data.cuda.assert_called_with(0)
        target.cuda.assert_called_with(0)

    def test_cuda_is_none(self):
        data = MagicMock()
        target = MagicMock()

        tuple_converter((data, target), None)

        data.cuda.assert_called_with()
        target.cuda.assert_called_with()
