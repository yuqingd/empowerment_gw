import numpy as np
from gym.utils import seeding
from gym import error, spaces, utils
import gym
from enum import IntEnum
from gym.envs.toy_text import discrete
from six import StringIO
import sys
from contextlib import closing

class GridWorldEnv(discrete.DiscreteEnv):

    class Actions(IntEnum):
        left = 0
        down = 1
        right = 2
        up = 3
        stay = 4

    def __init__(self, seed, grid_size, p, human_pos, boxes_pos, human_goal):
        """
        Gridworld environment with blocks.

        :param size: gridworld dimensions are (size, size)
        :param p:
        :param num_blocks:
        """

        self.num_boxes = int(len(boxes_pos)/2)
        assert self.num_boxes > 0, "Cannot have 0 Boxes"

        self.actions = GridWorldEnv.Actions
        self.action_dim = 2 # one for action, one for box number
        nA = len(self.actions) * self.num_boxes

        self.state_dim = 2 + self.num_boxes*2  #my coordinates, goal coordinates, and coordinates of boxes
        nS = grid_size ** self.state_dim
        self.grid_size = grid_size

        self.p = p

        self.seed(seed)

        isd = np.zeros(nS)

        self.cur_pos = human_pos
        self.boxes_pos = boxes_pos
        isd[self.to_s(np.concatenate((self.cur_pos, self.boxes_pos)))] = 1.0

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        #populate P
        for s in range(nS):
            for a in range(nA):
                li = P[s][a]
                if self.p == 1.0:
                    s_next = self.to_s(self.inc_boxes(self.from_s(s),self.from_a(a)))
                    li.append((1.0, s_next, 0, False)) # prob, next_s, rew, done
                else:
                    raise NotImplementedError

        self.human_goal = human_goal

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

    def inc_boxes(self, state_vec, a):
        """
        In
        :param state_vec:
        :param a:
        :return:
        """
        row, col = state_vec[0], state_vec[1]

        b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]
        b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]

        for cur_box in range(self.num_boxes):
            box, ac = a[1], a[0]
            if box is not cur_box:
                continue
            else:
                b_col = b_cols[box]
                b_row = b_rows[box]

                other_cols = np.copy(b_cols)
                other_cols[box] = col
                other_rows = np.copy(b_rows)
                other_rows[box] = row

                if ac == self.actions.left:
                    b_col = self.inc_(b_col, b_row, other_cols, other_rows, -1)
                elif ac == self.actions.down:
                    b_row = self.inc_(b_row, b_col, other_rows, other_cols, 1)
                elif ac == self.actions.right:
                    b_col = self.inc_(b_col, b_row, other_cols, other_rows, 1)
                elif ac == self.actions.up:
                    b_row = self.inc_(b_row, b_col, other_rows, other_cols, -1)
                elif ac == self.actions.stay:
                    pass

                b_cols[box] = b_col
                b_rows[box] = b_row

        return [row, col] + [*sum(zip(b_rows, b_cols), ())]

    def inc_(self, pos_move, pos_other, other_pos_move, other_pos_other, delta):
        target_block = False  # if target pos has a block or human, can't move block there
        for i in range(self.num_boxes):
            if (pos_move + delta, pos_other) == (other_pos_move[i], other_pos_other[i]):
                target_block = True
        if not target_block:
            pos_move = min(max(pos_move + delta, 0), self.grid_size - 1)
        return pos_move

    def to_s(self, positions):
        return np.sum([pos * (self.grid_size ** i) for i, pos in enumerate(positions)])

    def from_s(self, s):
        state_vec = []
        for i in range(self.state_dim):
            state_vec.append(s % self.grid_size)
            s //= self.grid_size
        return state_vec

    def to_a(self, action):
        return action[0] + action[1] * len(self.actions)

    def from_a(self, a):
        action_vec = []
        action_vec.append(a % len(self.actions))
        action_vec.append(a // len(self.actions))
        return action_vec

    def set_state(self, s):
        self.s = s

    def step_human(self, s):
        state_vec = self.from_s(s)
        row, col = state_vec[0], state_vec[1] #current human position

        b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)] # boxes rows
        b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)] # boxes cols

        best_row = None
        best_col = None
        dist = np.inf
        for ac in range(self.action_dim):
            if ac == self.actions.left:
                col = self.inc_(col, row, b_cols, b_rows, -1)
            elif ac == self.actions.down:
                row = self.inc_(row, col, b_rows, b_cols, 1)
            elif ac == self.actions.right:
                col = self.inc_(col, row, b_cols, b_rows, 1)
            elif ac == self.actions.up:
                row = self.inc_(row, col, b_rows, b_cols, -1)
            elif ac == self.actions.stay:
                pass

            # find the action that brings you closest to your goal
            cur_dist = np.linalg.norm(np.asarray([row, col]) - self.human_goal)

            if cur_dist < dist:
                dist = cur_dist
                best_row = row
                best_col = col

       # col = self.inc_(col, row, b_cols, b_rows, 1) #step left

        new_state = [best_row, best_col] + [*sum(zip(b_rows, b_cols), ())]

        return self.to_s(new_state)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        state_vec = self.from_s(self.s)
        row, col = state_vec[0], state_vec[1]
        b_rows = [state_vec[i] for i in range(2, self.state_dim - 1, 2)]
        b_cols = [state_vec[i] for i in range(3, self.state_dim, 2)]

        desc = [["0" for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        desc[row][col] = "1"
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        for box_row, box_col in zip(b_rows, b_cols):
            desc[box_row][box_col] = "2"
            desc[box_row][box_col] = utils.colorize(desc[box_row][box_col], "blue", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up","Stay"][self.lastaction % len(self.actions)]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
