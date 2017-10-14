class Plugin(object):
    def __call__(self, trainer):
        raise NotImplementedError()

    def prepare(self, trainer):
        pass
