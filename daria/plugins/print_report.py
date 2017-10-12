import sys
from ..plugin import AbstractPlugin


class PrintReport(AbstractPlugin):
    name = 'print_report'

    def __init__(self, entries, out=sys.stdout):
        self.entries = entries
        self.out = out
        entry_widths = [max(10, len(s)) for s in self.entries]
        self.header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *self.entries) + '\n'
        self.templates = []
        for entry, w in zip(self.entries, entry_widths):
            self.templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))

        self.latest_only = True

    def __call__(self, trainer):
        if self.header:
            self.out.write(self.header)
            self.header = None

        keys = trainer.history.keys()
        values = list(zip(*trainer.history.values()))
        if self.latest_only:
            values = [values[-1]]
        for value in values:
            self._print(dict(zip(keys, value)))

    def _print(self, info):
        for entry, template, empty in self.templates:
            if entry in info:
                self.out.write(template.format(info[entry]))
            else:
                self.out.write(empty)
        self.out.write('\n')
        self.out.flush()
