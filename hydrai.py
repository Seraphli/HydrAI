from util import RecentAvg
from replay import Replay, ReplayPack
from nn import NN
import numpy as np


class HydrAI(object):
    HEADS_N = 3

    def __init__(self):
        self.replay_size = 40
        self.baseline = RecentAvg(size=HydrAI.HEADS_N *
                                       self.replay_size, init=0)
        self.replays = {
            "good": ReplayPack(self.replay_size),
            "normal": ReplayPack(self.replay_size),
            "bad": ReplayPack(self.replay_size)
        }
        self.nns = {
            "good": NN(),
            "normal": NN(),
            "bad": NN()
        }

    def collect_replay(self):
        for _ in range(HydrAI.HEADS_N * self.replay_size):
            replay = self.play_one_game()
            self.baseline.update(replay.score)
            if replay.score > self.baseline.value + self.baseline.range * 0.25:
                self.replays["good"].add(replay)
            elif replay.score < self.baseline.value - self.baseline.range * 0.25:
                self.replays["bad"].add(replay)
            else:
                self.replays["normal"].add(replay)
        return self.baseline.value

    def play_one_game(self):
        replay = Replay()
        self.env.reset()
        while True:
            s = self.env.step()
            p_g = self.nns["good"].predict(s)
            p_n = self.nns["normal"].predict(s)
            p_b = self.nns["bad"].predict(s)
            p = p_g + p_n - p_b
            p /= np.sum(p)
            a = np.argmax(p)
            replay.add(s, a)
            if self.env.is_done():
                break
        return replay

    def train(self):
        loss = []
        loss.append(self.nns["good"].train(self.replays["good"].data))
        loss.append(self.nns["normal"].train(self.replays["normal"].data))
        loss.append(self.nns["bad"].train(self.replays["bad"].data))
        return loss
