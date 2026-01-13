class BaseALGO:
    def __init__(self, name):
        self.name = name
        self._observer = False
        self._compress = False

    def preprocess_weight(self):
        raise NotImplementedError

    def add_batch(self):
        raise NotImplementedError
