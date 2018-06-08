from util import RecentAvg, init_logger, get_run_timestamp
from replay import Replay, ReplayPack
from nn import NN
import numpy as np
from wrapper import wrap_deepmind
import gym
from functools import partial


class HydrAI(object):
    HEADS_N = 3

    def __init__(self):
        self.logger = init_logger(get_run_timestamp())
        self.env = gym.make("BreakoutNoFrameskip-v4")
        self.env = wrap_deepmind(self.env, frame_stack=True, scale=True)
        self.replay_size = 50
        # setup baseline
        # baseline will determine which replay pack the replay will be put into
        self.baseline = RecentAvg(size=HydrAI.HEADS_N *
                                       self.replay_size, init=0)
        self.baseline_mid = 0
        self.baseline_range = 0
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
            print(".", end="", flush=True)
            replay = self.play_one_game()
            self.logger.debug(replay.score)
            self.baseline.update(replay.score)
            collection.append(replay)
        print()
        if self.baseline_mid < self.baseline.value:
            self.baseline_mid = self.baseline.range / 2
        self.baseline_range = self.baseline.range
        self.logger.debug('range: {} {} {}'.format(
            self.baseline_mid - self.baseline_range * 0.25, self.baseline_mid,
            self.baseline_mid + self.baseline_range * 0.25))
        count = [0, 0, 0]
        for replay in collection:
            if replay.score >= self.baseline_mid + self.baseline_range * 0.25:
                self.replays["good"].add(replay)
                count[0] += 1
            elif replay.score <= self.baseline_mid - self.baseline_range * 0.25:
                self.replays["bad"].add(replay)
                count[2] += 1
            else:
                self.replays["normal"].add(replay)
                count[1] += 1
        self.logger.info('replay add {}'.format(count))
        self.logger.info('v: {}'.format(self.baseline.value))
        if count[2] > int(self.replay_size * 0.75):
            self.logger.warning('bad replay size larger than threshold')
        return self.baseline.value

    def play_one_game(self):
        replay = Replay()
        s = self.env.reset()
        while True:
            conv_s = np.reshape(s, [1, 84, 84, 4])
            p_g = self.nns["good"].predict(conv_s)
            self.logger.debug("p_g: {}".format(p_g["pi"][0]))
            p_n = self.nns["normal"].predict(conv_s)
            self.logger.debug("p_n: {}".format(p_n["pi"][0]))
            p_b = self.nns["bad"].predict(conv_s)
            self.logger.debug("p_b: {}".format(p_b["pi"][0]))
            p = 2 * p_g["pi"][0] + p_n["pi"][0] - p_b["pi"][0]
            self.logger.debug("p: {}".format(p))
            p += np.ones_like(self.a)
            p /= np.sum(p)
            self.logger.debug("p: {}".format(p))
            a = np.random.choice(self.a, p=p)
            self.logger.debug("a: {}".format(a))
            s_, r, t, _ = self.env.step(a)
            replay.add(s, a)
            replay.score += r
            s = s_
            if t:
                break
        return replay

    def train(self):
        self.logger.info("train")
        losses = []
        if len(self.replays["good"]) > 0:
            _loss = []
            for _ in range(self.replay_size * 5):
                _loss.append(self.nns["good"].train())
            losses.append(sum(_loss) / len(_loss))
        else:
            losses.append(0)
        if len(self.replays["normal"]) > 0:
            _loss = []
            for _ in range(self.replay_size * 5):
                _loss.append(self.nns["normal"].train())
            losses.append(sum(_loss) / len(_loss))
        else:
            losses.append(0)
        if len(self.replays["bad"]) > 0:
            _loss = []
            for _ in range(self.replay_size * 5):
                _loss.append(self.nns["bad"].train())
            losses.append(sum(_loss) / len(_loss))
        else:
            losses.append(0)
        self.logger.info('loss: {}'.format(losses))
        return losses
