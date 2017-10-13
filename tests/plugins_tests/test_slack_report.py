from __future__ import unicode_literals

import unittest
import mock
import io


from daria import plugins
from daria.plugins.slack_report import SlackIO


class TestSlackReport(unittest.TestCase):
    def setUp(self):
        self.slack_report = plugins.SlackReport(['time', 'epoch'], url=None)
        self.slack_report.out = io.StringIO()

        self.slack_io = SlackIO(url=None)
        self.slack_io.slack = mock.MagicMock()

    def test_slack_io_flush(self):
        slack_io = self.slack_io
        slack_io.write('poyo')
        slack_io.flush()

        slack_io.slack.notify.assert_called_once_with(
            text='```\npoyo```', username=slack_io.username,
            icon_emoji=':sushi:')

    def test_slack_report(self):
        slack_report = self.slack_report
        trainer_mock = mock.MagicMock()
        trainer_mock.history = {'time': [0], 'epoch': [0]}

        slack_report(trainer_mock)
