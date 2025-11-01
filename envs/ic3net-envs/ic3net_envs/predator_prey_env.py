#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
this file is for PP env of LEC_BEPAL model

Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision sqaure around it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
import copy

import random
import math
import curses

# 3rd party modules
import gym
import numpy as np
from gym import spaces
import random
from collections import defaultdict


class PredatorPreyEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self, ):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 1
        self.PREY_CLASS = 2
        self.PREDATOR_CLASS = 4
        self.GRID_CLASS = 3
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        self.episode_over = False
        self.map_dim = 4

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Prey Predator task')
        env.add_argument('--nenemies', type=int, default=1,
                         help="Total number of preys in play")
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box")
        env.add_argument('--vision', type=int, default=2,
                         help="Vision of predator")
        env.add_argument('--moving_prey', action="store_true", default=False,
                         help="Whether prey is fixed or moving")
        env.add_argument('--no_stay', action="store_true", default=False,
                         help="Whether predators have an action to stay in place")
        parser.add_argument('--mode', default='mixed', type=str,
                            help='cooperative|competitive|mixed (default: mixed)')
        env.add_argument('--enemy_comm', action="store_true", default=False,
                         help="Whether prey can communicate.")

    def multi_agent_init(self, args):

        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'moving_prey', 'mode', 'enemy_comm']
        for key in params:
            setattr(self, key, getattr(args, key))

        self.nprey = args.nenemies
        self.nprey = 1
        self.npredator = args.nfriendly
        self.dims = dims = (self.dim, self.dim)
        self.stay = not args.no_stay
        self.ngrid = args.obstacles

        if args.moving_prey:
            self.moving_prey = True
            # TODO

        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        # Define what an agent can do -
        if self.stay:
            self.naction = 5
        else:
            self.naction = 4

        self.action_space = spaces.MultiDiscrete([self.naction])

        self.BASE = (dims[0] * dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        self.GRID_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        # embed n*n*3
        self.self_explored = np.zeros([self.npredator, dims[0], dims[1]])
        self.others_explored = np.zeros([self.npredator, dims[0], dims[1]])
        self.agent_grid = np.zeros([ dims[0], dims[1]])
        self.prey_grid = np.zeros([dims[0], dims[1]])
        self.obstacle_grid = np.zeros([dims[0], dims[1]])

        self.agent_udt = np.zeros([self.npredator, 4, dims[0], dims[1]])
        self.ppweight = 2
        self.min_steps = 0
        self.comm = np.zeros([self.npredator])
        self.observed_obstacle = np.zeros(self.ngrid)
        # Setting max vocab size for 1-hot encoding
        self.vocab_size = self.BASE + 1 + 1 + 1 + self.npredator + 1
        #                   grid    + outside + prey + obstacle + predator

        # Observation for each agent will be vision * vision ndarray
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.vocab_size, self.dim, self.dim), dtype=int)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(self.vocab_size, (2 * self.vision) + 1, (2 * self.vision) + 1),
                                            dtype=int)
        # Actual observation will be of the shape 1 * npredator * (2v+1) * (2v+1) * vocab_size

        return

    def step(self, action):
        """
        The agents take a step in the environment.

        Parameters
        ----------
        action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :

            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")
        action = np.array(action).squeeze()
        action = np.atleast_1d(action)

        if self.moving_prey:
            self.prey_take_action()
        for i, a in enumerate(action):
            self._take_action(i, a)

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        self.set_explored()
        self.episode_over = False
        self.obs, action_mask = self._get_obs()
        
        debug = {'predator_locs': self.predator_loc, 'prey_locs': self.prey_loc}
        return self.obs, action_mask, self._get_reward(), self.episode_over, debug

    def reset(self, epoch):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.prey_captured = False
        self.team_captured = False
        self.reached_prey = np.zeros(self.npredator)

        self.self_explored = np.zeros([self.npredator, self.dims[0], self.dims[1]])
        self.others_explored = np.zeros([self.npredator, self.dims[0], self.dims[1]])
        self.agent_grid = np.zeros([self.dims[0], self.dims[1]])
        self.prey_grid = np.zeros([self.dims[0], self.dims[1]])
        self.obstacle_grid = np.zeros([self.dims[0], self.dims[1]])
        '''
        if (epoch // 100) < self.dim-1: #
            self.curr_gen_range = epoch // 100 +1
        else:'''
        self.curr_gen_range = self.dim

        # Locations
        locs = self._get_cordinates()  # original without obstacle
        wall_locs, locs = self.availiable_set()
        # self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]
        self.predator_loc, self.prey_loc = locs[:self.npredator], locs[self.npredator:]
        self.grid_loc = wall_locs


        self._set_grid()
        self.set_explored()
        # stat - like success ratio
        self.stat = dict()

        # Observation will be npredator * vision * vision ndarray
        self.obs, action_mask = self._get_obs()
        return self.obs, action_mask

    def seed(self):
        return

    def _get_cordinates(self):
        idx = np.random.choice(np.prod(self.dims), (self.npredator + self.nprey + self.ngrid), replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T

    def _set_grid(self):
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.grid[self.prey_loc[:,0], self.prey_loc[:,1]] = self.prey_ids

        # Padding for vision
        self.grid = np.pad(self.grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def set_explored(self):
        self.self_explored -= 0.1
        self.self_explored = np.clip(self.self_explored, 0, None)

        self.others_explored -= 0.1
        self.others_explored = np.clip(self.others_explored, 0, None)

        for i, p in enumerate(self.predator_loc):
            x_min = max(0, p[0] - self.vision)
            x_max = min(self.dims[0], p[0] + self.vision + 1)
            y_min = max(0, p[1] - self.vision)
            y_max = min(self.dims[1], p[1] + self.vision + 1)
            self.self_explored[i][x_min:x_max, y_min:y_max] = 1
            for j, q in enumerate(self.predator_loc):
                if i == j:
                    continue
                self.others_explored[j][x_min:x_max, y_min:y_max] = 1

    def get_cnn_observation(self):
        """
        Return CNN-style full-grid observation for each predator agent.

        Output shape: [N, L, M, M]
        Where:
            - N: number of agents
            - L: number of channels (agent, prey, obstacle, explored)
            - M: map size
        """
        M = self.dims[0]
        L = 4  # [agent, prey, obstacle, explored]
        obs_tensor = np.zeros((self.npredator, L, M, M), dtype=np.float32)

        for i, pos in enumerate(self.predator_loc):
            # Channel 0: Agent positions
            obs_tensor[i, 0] = self.agent_grid
            obs_tensor[i, 0, pos[0], pos[1]] = 1  # emphasize own position

            # Channel 1: Prey positions
            obs_tensor[i, 1] = self.prey_grid

            # Channel 2: Obstacle positions
            obs_tensor[i, 2] = self.obstacle_grid

            # Channel 3: Explored area
            obs_tensor[i, 3] = self.self_explored[i]

            # Mask outside vision range
            x_min = max(0, pos[0] - self.vision)
            x_max = min(M, pos[0] + self.vision + 1)
            y_min = max(0, pos[1] - self.vision)
            y_max = min(M, pos[1] + self.vision + 1)

            mask = np.zeros((M, M), dtype=bool)
            mask[x_min:x_max, y_min:y_max] = True

            for l in range(L):
                obs_tensor[i, l][~mask] = 0

        return obs_tensor  # shape: [N, L, M, M]

    def _get_obs(self):

        self.bool_base_grid = self.empty_bool_base_grid.copy()
        self.agent_grid = np.zeros([self.dims[0], self.dims[1]])
        self.prey_grid = np.zeros([self.dims[0], self.dims[1]])
        self.obstacle_grid = np.zeros([self.dims[0], self.dims[1]])
        for i, p in enumerate(self.predator_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS + i] += 1
            self.agent_grid[p[0],p[1]] += 1

        for i, p in enumerate(self.prey_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1
            self.prey_grid[p[0], p[1]] += 1

        for i, p in enumerate(self.grid_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.GRID_CLASS] += 1
            self.obstacle_grid[p[0], p[1]] += 1

        obs = []
        action_mask = []
        for p in self.predator_loc:
            # slice_y = slice(self.vision, self.dim+self.vision)
            # slice_x = slice(self.vision, self.dim+self.vision)
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            obs.append(self.bool_base_grid[slice_y, slice_x])
            # Build action mask
            mask = np.ones(self.naction, dtype=int)
            if not self.is_valid_move(p.tolist(), 0): mask[0] = 0  # UP
            if not self.is_valid_move(p.tolist(), 1): mask[1] = 0  # RIGHT
            if not self.is_valid_move(p.tolist(), 2): mask[2] = 0  # DOWN
            if not self.is_valid_move(p.tolist(), 3): mask[3] = 0  # LEFT
            # If staying is not allowed (according to args), mask it out
            if not self.stay: mask[4] = 0
            action_mask.append(mask)
            

        if self.enemy_comm:
            for p in self.prey_loc:
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                obs.append(self.bool_base_grid[slice_y, slice_x])
                action_mask = np.vstack([action_mask, np.ones(self.naction, dtype=int)])

                
        obs = np.stack(obs)
        action_mask = np.stack(action_mask)
        # obs = obs[:, :, :, -(1+ 1+1 + self.npredator + 1):]
        return obs, action_mask

    def _take_action(self, idx, act):
        # prey action
        if idx >= self.npredator:
            # fixed prey
            if not self.moving_prey:
                return
            else:
                raise NotImplementedError

        if self.reached_prey[idx] == 1:
            return

        # STAY action
        if act == 5:
            return

        location = copy.deepcopy(self.predator_loc[idx])
        if act == 0:
            location[0] = location[0] - 1
        elif act == 1:
            location[1] = location[1] + 1
        elif act == 2:
            location[0] = location[0] + 1
        elif act == 3:
            location[1] = location[1] - 1

        # UP
        if act == 0 and self.grid[max(0,
                                      self.predator_loc[idx][0] + self.vision - 1),
        self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS \
                and location.tolist() not in self.grid_loc.tolist():
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0] - 1)

        # RIGHT
        elif act == 1 and self.grid[self.predator_loc[idx][0] + self.vision,
        min(self.dims[1] - 1,
            self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS \
                and location.tolist() not in self.grid_loc.tolist():
            self.predator_loc[idx][1] = min(self.dims[1] - 1,
                                            self.predator_loc[idx][1] + 1)

        # DOWN
        elif act == 2 and self.grid[min(self.dims[0] - 1,
                                        self.predator_loc[idx][0] + self.vision + 1),
        self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS \
                and location.tolist() not in self.grid_loc.tolist():
            self.predator_loc[idx][0] = min(self.dims[0] - 1,
                                            self.predator_loc[idx][0] + 1)

        # LEFT
        elif act == 3 and self.grid[self.predator_loc[idx][0] + self.vision,
        max(0,
            self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS \
                and location.tolist() not in self.grid_loc.tolist():
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1] - 1)

    def _get_reward(self):
        n = self.npredator if not self.enemy_comm else self.npredator + self.nprey
        reward = np.full(n, self.TIMESTEP_PENALTY)

        on_prey = np.where(np.all(self.predator_loc == self.prey_loc, axis=1))[0]
        nb_predator_on_prey = on_prey.size
        # If at least one predator is on the prey, mark it as captured
        if nb_predator_on_prey > 0:
            self.prey_captured = True
        
        if self.mode == 'cooperative':
            reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'competitive':
            if nb_predator_on_prey:
                reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
        elif self.mode == 'mixed':
            reward[on_prey] = self.PREY_REWARD
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        self.reached_prey[on_prey] = 1

        if np.all(self.reached_prey == 1) and self.mode == 'mixed':
            self.episode_over = True

        # Prey reward
        if nb_predator_on_prey == 0:
            reward[self.npredator:] = -1 * self.TIMESTEP_PENALTY
        else:
            # TODO: discuss & finalise
            reward[self.npredator:] = 0

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.npredator:
                self.team_captured = True
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0

        return reward

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _onehot_initialization(self, a):
        ncols = self.vocab_size
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.stdscr.clear()

        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'

        for p in self.prey_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(3))
                    elif 'X' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '0'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def seedset(self):
        pos = [-1, 1, 0]
        a = random.choice(pos)
        b = random.choice(pos)
        return a, b

    def exit_render(self):
        curses.endwin()

    def prey_take_action(self):
        if self.prey_captured:  # <-- ADD THIS CHECK
            return
        best_action = self.escape()

        location = copy.deepcopy(self.prey_loc[0])
        move_flag = False
        for act in best_action:
            if move_flag:
                break
            if act == 0:
                location[0] = location[0] - 1
            elif act == 1:
                location[1] = location[1] + 1
            elif act == 2:
                location[0] = location[0] + 1
            elif act == 3:
                location[1] = location[1] - 1

            if act == 5:
                return
            # UP
            if act == 0 and self.grid[max(0,
                                          self.prey_loc[0][0] + self.vision - 1),
            self.prey_loc[0][1] + self.vision] != self.OUTSIDE_CLASS \
                    and location.tolist() not in self.grid_loc.tolist():
                self.prey_loc[0][0] = max(0, self.prey_loc[0][0] - 1)
                move_flag = True

                # RIGHT
            elif act == 1 and self.grid[self.prey_loc[0][0] + self.vision,
            min(self.dims[1] - 1,
                self.prey_loc[0][1] + self.vision + 1)] != self.OUTSIDE_CLASS \
                    and location.tolist() not in self.grid_loc.tolist():
                self.prey_loc[0][1] = min(self.dims[1] - 1,
                                          self.prey_loc[0][1] + 1)
                move_flag = True

            # DOWN
            elif act == 2 and self.grid[min(self.dims[0] - 1,
                                            self.prey_loc[0][0] + self.vision + 1),
            self.prey_loc[0][1] + self.vision] != self.OUTSIDE_CLASS \
                    and location.tolist() not in self.grid_loc.tolist():
                self.prey_loc[0][0] = min(self.dims[0] - 1,
                                          self.prey_loc[0][0] + 1)
                move_flag = True

            # LEFT
            elif act == 3 and self.grid[self.prey_loc[0][0] + self.vision,
            max(0,
                self.prey_loc[0][1] + self.vision - 1)] != self.OUTSIDE_CLASS \
                    and location.tolist() not in self.grid_loc.tolist():
                self.prey_loc[0][1] = max(0, self.prey_loc[0][1] - 1)
                move_flag = True

    # def escape(self):  # mme
    #     keys = [0, 1, 2, 3, 5]  # UP, RIGHT, DOWN, LEFT, STAY
    #     available_actions = dict.fromkeys(keys, 0)
    #     x, y = self.prey_loc[0]
    
    #     blocked_directions = set()  # Track blocked directions
    
    #     # Check if moving out of the map
    #     if x == 0:  # At top boundary, UP is blocked
    #         blocked_directions.add(0)
    #     if x == self.dims[0] - 1:  # At bottom boundary, DOWN is blocked
    #         blocked_directions.add(2)
    #     if y == 0:  # At left boundary, LEFT is blocked
    #         blocked_directions.add(3)
    #     if y == self.dims[1] - 1:  # At right boundary, RIGHT is blocked
    #         blocked_directions.add(1)
    
    #     for p in self.predator_loc:
    #         distance = abs(x - p[0]) + abs(y - p[1])  # Manhattan Distance
    #         if abs(x - p[0]) > self.vision or abs(y - p[1]) > self.vision:
    #             continue  # Ignore predators too far away
    
    #         weight = 1 / (distance + 1)  # Closer predators are more dangerous
    
    #         # Predator is below, prey wants to go UP (0)
    #         if p[0] - x > 0:
    #             available_actions[0] += weight
    #             if p[0] - x == 1 and p[1] == y:  # Predator directly below → block DOWN (2)
    #                 blocked_directions.add(2)
    
    #         # Predator is above, prey wants to go DOWN (2)
    #         elif x - p[0] > 0:
    #             available_actions[2] += weight
    #             if x - p[0] == 1 and p[1] == y:  # Predator directly above → block UP (0)
    #                 blocked_directions.add(0)
    
    #         # Predator is right, prey wants to go LEFT (3)
    #         if p[1] - y > 0:
    #             available_actions[3] += weight
    #             if p[1] - y == 1 and p[0] == x:  # Predator directly right → block RIGHT (1)
    #                 blocked_directions.add(1)
    
    #         # Predator is left, prey wants to go RIGHT (1)
    #         elif y - p[1] > 0:
    #             available_actions[1] += weight
    #             if y - p[1] == 1 and p[0] == x:  # Predator directly left → block LEFT (3)
    #                 blocked_directions.add(3)
    
    #     # Check for wall obstacles blocking movement
    #     for direction in [0, 1, 2, 3]:
    #         if not self.is_valid_move([x, y], direction):
    #             blocked_directions.add(direction)
    
    #     # If at least three directions are blocked, avoid staying
    #     if len(blocked_directions) >= 3:
    #         available_actions[5] = -999  # Avoid staying
    #     # for invalid_action in list(blocked_directions):
    #     #     available_actions[invalid_action] = -999
    #     # if sum(available_actions.values()) == 0:
    #     #     available_actions[5] = 999
    
    #     # Sort actions by least danger
    #     sorted_actions = sorted(available_actions.items(), key=lambda item: item[1], reverse=True)
    
    #     # Filter valid moves (avoid obstacles and map boundaries)
    #     valid_moves = [act for act, v in sorted_actions if act not in blocked_directions and v > 0]
    
    #     return valid_moves if valid_moves else [5]  # If no valid moves, stay

    def escape(self): # mmss
        keys = [0, 1, 2, 3, 5]  # UP, RIGHT, DOWN, LEFT, STAY
        action_scores = {k: 0 for k in keys}
        x, y = self.prey_loc[0]

        # --- Step 1: Score actions based on predator locations ---
        # A higher score will mean a SAFER direction.
        for p in self.predator_loc:
            # Ignore predators outside of vision range
            if abs(x - p[0]) > self.vision or abs(y - p[1]) > self.vision:
                continue

            distance = abs(x - p[0]) + abs(y - p[1])

            # If predator is directly adjacent, it's a huge threat.
            # Heavily penalize moving towards it.
            if distance == 1:
                if p[0] - x > 0: action_scores[2] -= 1000  # Predator below, penalize DOWN
                if x - p[0] > 0: action_scores[0] -= 1000  # Predator above, penalize UP
                if p[1] - y > 0: action_scores[1] -= 1000  # Predator right, penalize RIGHT
                if y - p[1] > 0: action_scores[3] -= 1000  # Predator left, penalize LEFT

            # Prefer directions that increase distance to the predator.
            # The further the predator, the smaller the influence.
            weight = 1 / (distance + 1e-6)  # Add epsilon to avoid division by zero

            if p[0] > x:
                action_scores[0] += weight  # Predator below -> UP is safer
            elif p[0] < x:
                action_scores[2] += weight  # Predator above -> DOWN is safer
            if p[1] > y:
                action_scores[3] += weight  # Predator right -> LEFT is safer
            elif p[1] < y:
                action_scores[1] += weight  # Predator left -> RIGHT is safer

        # --- Step 2: Make impossible moves have an infinitely bad score ---
        # Check for walls and map boundaries
        for direction in [0, 1, 2, 3]:
            if not self.is_valid_move([x, y], direction):
                action_scores[direction] = -float('inf')

        # If the prey is cornered, staying still is a very bad idea.
        num_blocked = sum(1 for score in action_scores.values() if score == -float('inf'))
        if num_blocked >= 2:  # If at least two directions are hard-blocked
            action_scores[5] -= 500  # Penalize staying

        # --- Step 3: Choose the best of the valid moves ---
        sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)

        # Filter out any moves that are impossible (into walls)
        valid_moves = [act for act, score in sorted_actions if score != -float('inf')]

        # If all options are impossible, the prey has no choice but to stay
        if not valid_moves:
            return [5]

        # Return all valid actions, sorted from best to worst.
        # The prey_take_action function will try them in order.
        return valid_moves

    # def escape(self):   # dl 5: mmss+stay
    #     keys = [0, 1, 2, 3, 5]  # UP, RIGHT, DOWN, LEFT, STAY
    #     action_scores = {k: 0 for k in keys}
    #     x, y = self.prey_loc[0]
    #
    #     # Check if any predators are within the vision range
    #     predators_in_sight = [
    #         p for p in self.predator_loc
    #         if abs(x - p[0]) <= self.vision*2 and abs(y - p[1]) <= self.vision*2
    #     ]
    #
    #     # --- Step 1: Handle the "stay" case first if no predators are nearby ---
    #     if not predators_in_sight:
    #         return [5]  # The prey should stay still if it's not being observed
    #
    #     # If predators are in sight, proceed with the original logic
    #     # --- Step 2: Score actions based on predator locations ---
    #     # A higher score will mean a SAFER direction.
    #     for p in predators_in_sight:
    #         distance = abs(x - p[0]) + abs(y - p[1])
    #
    #         # If predator is directly adjacent, it's a huge threat.
    #         # Heavily penalize moving towards it.
    #         if distance == 1:
    #             if p[0] - x > 0: action_scores[2] -= 1000  # Predator below, penalize DOWN
    #             if x - p[0] > 0: action_scores[0] -= 1000  # Predator above, penalize UP
    #             if p[1] - y > 0: action_scores[1] -= 1000  # Predator right, penalize RIGHT
    #             if y - p[1] > 0: action_scores[3] -= 1000  # Predator left, penalize LEFT
    #
    #         # Prefer directions that increase distance to the predator.
    #         # The further the predator, the smaller the influence.
    #         weight = 1 / (distance + 1e-6)  # Add epsilon to avoid division by zero
    #
    #         if p[0] > x:
    #             action_scores[0] += weight  # Predator below -> UP is safer
    #         elif p[0] < x:
    #             action_scores[2] += weight  # Predator above -> DOWN is safer
    #         if p[1] > y:
    #             action_scores[3] += weight  # Predator right -> LEFT is safer
    #         elif p[1] < y:
    #             action_scores[1] += weight  # Predator left -> RIGHT is safer
    #
    #     # --- Step 3: Make impossible moves have an infinitely bad score ---
    #     # Check for walls and map boundaries
    #     for direction in [0, 1, 2, 3]:
    #         if not self.is_valid_move([x, y], direction):
    #             action_scores[direction] = -float('inf')
    #
    #     # If the prey is cornered, staying still is a very bad idea.
    #     num_blocked = sum(1 for score in action_scores.values() if score == -float('inf'))
    #     if num_blocked >= 2:  # If at least two directions are hard-blocked
    #         action_scores[5] -= 500  # Penalize staying
    #
    #     # --- Step 4: Choose the best of the valid moves ---
    #     sorted_actions = sorted(action_scores.items(), key=lambda item: item[1], reverse=True)
    #
    #     # Filter out any moves that are impossible (into walls)
    #     valid_moves = [act for act, score in sorted_actions if score != -float('inf')]
    #
    #     # If all options are impossible, the prey has no choice but to stay
    #     if not valid_moves:
    #         return [5]
    #
    #     # Return all valid actions, sorted from best to worst.
    #     # The prey_take_action function will try them in order.
    #     return valid_moves

    def is_valid_move(self, position, action):
        """ Check if a move is valid (not blocked by a wall or out of bounds) """
        new_position = position.copy()

        if action == 0 and position[0] > 0:  # UP
            new_position[0] -= 1
        elif action == 1 and position[1] < self.dims[1] - 1:  # RIGHT
            new_position[1] += 1
        elif action == 2 and position[0] < self.dims[0] - 1:  # DOWN
            new_position[0] += 1
        elif action == 3 and position[1] > 0:  # LEFT
            new_position[1] -= 1
        else:
            return False  # Out of bounds move
        if new_position in self.grid_loc.tolist():
            return False
        else:
            return True

    def availiable_set(self):
        wall_grids, avaliable_grids = get_wall_grids(self.dim, self.ngrid)
        idx = np.random.choice(range(len(avaliable_grids)), (self.nprey), replace=False)  # self.npredator +
        overlap_agent_locs = []
        curr_range = self.curr_gen_range - 1
        while len(overlap_agent_locs) < self.npredator:
            curr_range += 1
            overlap_agent_locs = agent_init_range(avaliable_grids, idx, curr_range)
        agent_idx = np.random.choice(range(len(overlap_agent_locs)), (self.npredator), replace=False)
        agent_locs = np.array(overlap_agent_locs)[agent_idx]
        prey_locs = np.array(avaliable_grids[idx[0]])
        locs = np.vstack((agent_locs, prey_locs))
        wall = [item for sublist in wall_grids for item in sublist]
        return np.array(wall), np.array(locs)


def generate_random_wall_from_dicts(column_dict, row_dict, obstacle_limit):
    wall_length = random.randint(3, int(obstacle_limit / 2))
    orientation = random.choice(['horizontal', 'vertical'])

    if orientation == 'horizontal':
        row = random.choice(list(row_dict.keys()))

        if len(row_dict[row]) >= wall_length:
            start_index = random.randint(0, len(row_dict[row]) - wall_length)
            wall_coordinates = [row_dict[row][start_index + col] for col in range(wall_length)]
        else:
            return None  # Not enough space to place the wall

    else:  # vertical
        col = random.choice(list(column_dict.keys()))

        if len(column_dict[col]) >= wall_length:
            start_index = random.randint(0, len(column_dict[col]) - wall_length)
            wall_coordinates = [column_dict[col][start_index + row] for row in range(wall_length)]
        else:
            return None  # Not enough space to place the wall

    return wall_coordinates


def get_adjacent_coordinates(wall_coordinates):
    adjacent_coordinates = set()
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

    for x, y in wall_coordinates:
        for dx, dy in directions:
            adjacent_x = x + dx
            adjacent_y = y + dy
            # Add adjacent coordinates as tuples
            adjacent_coordinates.add((adjacent_x, adjacent_y))

    return adjacent_coordinates


def remove_coordinates_from_dicts(column_dict, row_dict, wall_coordinates):
    # Convert wall coordinates to a set of tuples for efficient checking
    occupied_coordinates = set(tuple(coord) for coord in wall_coordinates) | get_adjacent_coordinates(
        wall_coordinates)

    # Remove from column dictionary
    for col in column_dict:
        column_dict[col] = [coord for coord in column_dict[col] if tuple(coord) not in occupied_coordinates]

    # Remove from row dictionary
    for row in row_dict:
        row_dict[row] = [coord for coord in row_dict[row] if tuple(coord) not in occupied_coordinates]


# Main function to run the complete process
def get_wall_grids(map_size, obstacle_limit):
    column_dict = {col: [] for col in range(map_size)}
    row_dict = {row: [] for row in range(map_size)}

    for x in range(map_size):
        for y in range(map_size):
            coordinate = [x, y]
            column_dict[y].append(coordinate)
            row_dict[x].append(coordinate)

    total_obstacles = 0
    wall_coordinates_list = []  # List to hold all wall coordinates

    while total_obstacles < obstacle_limit:
        wall_coords = generate_random_wall_from_dicts(column_dict, row_dict, obstacle_limit)

        if wall_coords:
            wall_coordinates_list.append(wall_coords)  # Store the generated wall coordinates
            # print(f"Generated wall coordinates: {wall_coords}")
            # Remove wall and adjacent coordinates from dictionaries
            remove_coordinates_from_dicts(column_dict, row_dict, wall_coords)
            total_obstacles += len(wall_coords)
            # print(f"Remaining available coordinates in column dict: {[len(v) for v in column_dict.values()]}")
            # print(f"Remaining available coordinates in row dict: {[len(v) for v in row_dict.values()]}")
        else:
            # print("Not enough space to place a new wall, stopping generation.")
            break  # Stop if not enough space to place a wall

    # Collect available coordinates from both dictionaries
    available_coordinates = [coord for col in column_dict for coord in column_dict[col]]

    # print(f"Total obstacles added: {total_obstacles}", flush=True)
    return wall_coordinates_list, available_coordinates


def agent_init_range(avaliable_grids, idx, dim):
    avaliable_grids_raw = []
    for dx in range(-dim, dim + 1):  # Include both ends of the range
        for dy in range(-dim, dim + 1):
            new_x = avaliable_grids[idx[0]][0] + dx
            new_y = avaliable_grids[idx[0]][1] + dy
            avaliable_grids_raw.append([new_x, new_y])
    set_available_grids = {tuple(coord) for coord in avaliable_grids}

    # Find overlapping coordinates in available_grids_for_dims
    overlap = [coord for coord in avaliable_grids_raw if tuple(coord) in set_available_grids]

    return overlap


