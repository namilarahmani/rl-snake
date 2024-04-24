import torch
import numpy as np
from random import randint


def baseline_action(state):
    # baseline action has no dependence on the old state as it just returns a random action
    return randint(0, 2)