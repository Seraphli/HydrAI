from hydrai import HydrAI

max_iter = 1000

ai = HydrAI()
for _ in range(max_iter):
    print(ai.collect_replay())
    print(ai.train())
