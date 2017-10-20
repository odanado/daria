from __future__ import unicode_literals

import os
import io
import slackweb

from . import PrintReporter


class SlackIO(io.StringIO):
    def __init__(self, url, username=None, *args, **kwargs):
        super(SlackIO, self).__init__(*args, **kwargs)
        if username is None:
            self.username = os.uname()[1]
        else:
            self.username = username
        self.slack = slackweb.Slack(url=url)

    def flush(self, *args, **kwargs):
        super(SlackIO, self).flush(*args, **kwargs)
        text = "```\n{}```".format(self.getvalue())
        self.slack.notify(text=text,
                          username=self.username, icon_emoji=":sushi:")


class SlackReporter(PrintReporter):
    name = 'slack_reporter'

    def __init__(self, entries, url, username=None):
        self.out = SlackIO(url, username=username)
        super(SlackReporter, self).__init__(entries, out=self.out)

        self.header_org = self.header

    def __call__(self, trainer):
        self.header = self.header_org
        super(SlackReporter, self).__call__(trainer)
        self.out.truncate(0)
