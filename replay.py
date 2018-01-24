import numpy as np
import random


class Replay(object):
    def __init__(self, size=0):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.score = 0

    def __len__(self):
        return len(self._storage)

    def add(self, s, a):
        data = (s, a)

        if self._maxsize == 0:
            self._storage.append(data)
            return

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        _s, _a = [], []
        for i in idxes:
            data = self._storage[i]
            s, a, r, t, s_ = data
            _s.append(np.array(s, copy=False))
            _a.append(np.array(a, copy=False))
        return np.array(_s), np.array(_a)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class ReplayPack(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def add(self, replay):
        if self._next_idx >= len(self._storage):
            self._storage.append(replay)
        else:
            self._storage[self._next_idx] = replay
        self._next_idx = (self._next_idx + 1) % self._maxsize
