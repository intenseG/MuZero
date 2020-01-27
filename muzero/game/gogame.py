from typing import List
import numpy as np

import gym

from game.game import Action, AbstractGame
from game.gym_wrappers import ScalingObservationWrapper
from gym_go import govars
# from gym_go.gogame import GoGame as gym_gogame


class GoGame(AbstractGame):
    """The Gym GoGame environment"""

    def __init__(self, discount: float, size: int, reward_method='real'):
        super().__init__(discount)
        self.env = gym.make('gym_go:go-v0', size=size, reward_method=reward_method)
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.observations = [self.env.reset()]
        self.done = False

    @property
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return len(self.actions)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""

        if isinstance(action, Action):
            observation, reward, done, _ = self.env.step(action.index)
        elif isinstance(action, int):
            observation, reward, done, _ = self.env.step(action)
        self.observations += [observation]
        self.done = done
        return reward

    def terminal(self) -> bool:
        """Is the game is finished?"""
        return self.done

    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        valid_moves = self.env.get_valid_moves()
        valid_move_idcs = np.argwhere(valid_moves).flatten()
        # print(valid_moves)
        # print(valid_move_idcs)
        # print(len(valid_moves))
        # print(len(valid_move_idcs))
        self.env.get_children(canonical=True)
        # non_pass_idcs = np.where(valid_move_idcs != self.env.size * self.env.size)
        # legal_cnt = len(non_pass_idcs)
        self.actions.clear()
        for i in range(len(valid_move_idcs) - 1):
            self.actions.append(Action(valid_move_idcs[i]))
            # print(i)
            # y = i // self.env.size
            # x = i % self.env.size
            # print(f'({y}, {x})')
            # if y == self.env.size or x == self.env.size:
            #     self.actions.append(Action(i))
            # elif valid_move_idcs[i] != 1:
            #     self.actions.append(Action(i))
        # for a in self.actions:
        #     print(f'[after] {a.index}')
        # print(f'[gogame] Actions length: {len(self.actions)}')
        return self.actions

    def make_image(self, state_index: int):
        """Compute the state of the game."""
        return self.observations[state_index]

    def show_board(self, action=None):
        self.env.render(action=action)
