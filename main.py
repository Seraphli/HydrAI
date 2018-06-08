from hydrai import HydrAI

max_iter = 1000

ai = HydrAI()
for _ in range(max_iter):
    # first we need to collect replays
    baseline_value = ai.collect_replay()
    # then we train ai on their certain replays
    loss = ai.train()
