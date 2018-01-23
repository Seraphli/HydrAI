class Replay(object):
    """docstring for Replay"""

    def __init__(self, arg):
        super(Replay, self).__init__()
        self.arg = arg

    def update(self, record, terminal):
        # add record
        if terminal:
            self.score = record.score


def play_one_game():
    # intergrate with environment
    return Replay()


baseline = Average(init=0)

good = queue(max=20)
normal = queue(max=20)
bad = queue(max=20)

for _ in range(MAX_ITER):
    replay = play_one_game()
    baseline.update(replay.score)
    if replay.score > baseline.value + baseline.range * 0.25:
        good.append(replay)
    elif replay.score < baseline.value - baseline.range * 0.25:
        bad.append(replay)
    else:
        normal.append(replay)


