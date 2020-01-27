"""Helpers for the MCTS"""
from typing import Optional

import numpy as np

MAXIMUM_FLOAT_VALUE = float('inf')


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        if value is None:
            raise ValueError

        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        # If the value is unknow, by default we set it to the minimum possible value
        if value is None:
            return 0.0

        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):
    """A class that represent nodes inside the MCTS tree"""

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> Optional[float]:
        if self.visit_count == 0:
            return None
        return self.value_sum / self.visit_count


def softmax_sample(visit_counts, actions, t):
    # print(f'[utils] Actions type: {type(actions)}')
    # print(f'[utils] Actions length: {len(actions)}')
    # action_list = [a.index for a in actions]
    # c = np.max(action_list)
    # exp_array = np.exp(action_list - c)
    # sum_exp_array = np.sum(exp_array)
    # y = exp_array / sum_exp_array
    # print(f'y: {y}')
    counts_exp = np.exp(visit_counts) * (1 / t)
    probs = counts_exp / np.sum(counts_exp, axis=0)
    # if np.isnan(probs):
    # print(f'visit_counts: {visit_counts}')
    # print(f'counts_exp: {counts_exp}')
    # print(f'np.sum(counts_exp, axis=0): {np.sum(counts_exp, axis=0)}')
    # print(f'probs: {probs}')
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]
