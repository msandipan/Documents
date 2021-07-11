import time
from ray import tune
import os
import json

def train_func(config, checkpoint_dir=None):
    start = 0
    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            state = json.loads(f.read())
            start = state["step"] + 1

    for iter in range(start, 100):
        time.sleep(1)

        with tune.checkpoint_dir(step=step) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            with open(path, "w") as f:
                f.write(json.dumps({"step": start}))

        tune.report(hello="world", ray="tune")

tune.run(train_func)