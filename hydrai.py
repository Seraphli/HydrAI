from util import RecentAvg
from replay import Replay, ReplayPack
from nn import NN
import numpy as np
from wrapper import wrap_deepmind
import gym
from functools import partial


class HydrAI(object):
    HEADS_N = 3

    def __init__(self):
        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.env = wrap_deepmind(self.env, frame_stack=True, scale=True)
        self.replay_size = 40
        # setup baseline
        # baseline will determine which replay pack the replay will be put into
        self.baseline = RecentAvg(size=HydrAI.HEADS_N *
                                       self.replay_size, init=0)
        self.replays = {
            "good": ReplayPack(self.replay_size),
            "normal": ReplayPack(self.replay_size),
            "bad": ReplayPack(self.replay_size)
        }
        feature_size = self.env.observation_space.shape
        action_size = self.env.action_space.n
        self.nns = {
            "good": NN(feature_size, action_size,
                       [partial(self.replays["good"].sample, 32)],
                       "good_"),
            "normal": NN(feature_size, action_size,
                         [partial(self.replays["normal"].sample, 32)],
                         "normal_"),
            "bad": NN(feature_size, action_size,
                      [partial(self.replays["bad"].sample, 32)],
                      "bad_")
        }
        self.a = list(range(action_size))

    def collect_replay(self):
        collection = []
        for _ in range(self.replay_size):
            print("playing {}".format(_), end="")
            replay = self.play_one_game()
            self.baseline.update(replay.score)
            collection.append(replay)
        for replay in collection:
            if replay.score > self.baseline.value + self.baseline.range * 0.25:
                self.replays["good"].add(replay)
            elif replay.score < self.baseline.value - self.baseline.range * 0.25:
                self.replays["bad"].add(replay)
            else:
                self.replays["normal"].add(replay)
        return self.baseline.value

    def play_one_game(self):
        replay = Replay()
        s = self.env.reset()
        count = 0
        while True:
            conv_s = np.reshape(s, [1, 84, 84, 4])
            p_g = self.nns["good"].predict(conv_s)
            p_n = self.nns["normal"].predict(conv_s)
            p_b = self.nns["bad"].predict(conv_s)
            p = 2 * p_g["pi"][0] + p_n["pi"][0] - p_b["pi"][0]
            p += np.ones_like(self.a)
            p /= np.sum(p)
            a = np.random.choice(self.a, p=p)
            s_, r, t, _ = self.env.step(a)
            replay.add(s, a)
            replay.score += r
            s = s_
            count += 1
            if count % 10 == 0:
                print(".", end="", flush=True)
            if t:
                print()
                break
        return replay

    def train(self):
        losses = []
        if len(self.replays["good"]) > 0:
            _loss = []
            for _ in range(self.replay_size):
                _loss.append(self.nns["good"].train())
            losses.append(sum(_loss) / len(_loss))
        else:
            losses.append(0)
        if len(self.replays["normal"]) > 0:
            _loss = []
            for _ in range(self.replay_size):
                _loss.append(self.nns["normal"].train())
            losses.append(sum(_loss) / len(_loss))
        else:
            losses.append(0)
        if len(self.replays["bad"]) > 0:
            _loss = []
            for _ in range(self.replay_size):
                _loss.append(self.nns["bad"].train())
            losses.append(sum(_loss) / len(_loss))
        else:
            losses.append(0)
        return losses
