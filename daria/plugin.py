class AbstractPlugin(object):
    def __call__(self, trainer):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self, state):
        raise NotImplementedError()
