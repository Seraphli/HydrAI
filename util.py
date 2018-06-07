def get_path(name):
    import os
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), name))
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


class RecentAvg(object):
    def __init__(self, size=50, init=None):
        self._size = size
        self._current = 0
        if init is not None:
            self._value = init
            self._values = [init for _ in range(size)]
        else:
            self._value = None
            self._values = []

    def update(self, new_val):
        if len(self._values) == self._size:
            self._values[self._current] = new_val
        else:
            self._values.append(new_val)
        self._current = (self._current + 1) % self._size
        self._value = sum(self._values) / len(self._values)

    @property
    def value(self):
        return self._value

    @property
    def max(self):
        return max(self._values)

    @property
    def min(self):
        return min(self._values)

    @property
    def range(self):
        return self.max - self.min

    def __repr__(self):
        return str(self.value)
