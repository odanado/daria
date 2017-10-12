class Plugin(object):
    def __call__(self, trainer):
        raise NotImplementedError()
