from util import RecentAvg

baseline = RecentAvg(init=0)

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