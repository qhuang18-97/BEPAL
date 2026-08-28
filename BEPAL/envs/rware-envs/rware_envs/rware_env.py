import time
import numpy as np
import gym
# from gym.utils.save_video import save_video
import rware
import torch
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete


class RwareEnv():

    def __init__(self, args):
        request_queue_size = 4

        # Create the environment
        # Either: tiny, small, medium, large
        # Can change number of agents (i.e. '4ag'), queue size is equal to the number of agents
        # Can change layout using layout = SOME_LAYOUT_STRING
        env = gym.make("rware-tiny-4ag-v1", request_queue_size=request_queue_size, max_steps=args.max_steps)

        env.reset()
        n_agents = env.n_agents
        
        observation_sizes = self.extract_sizes(env.observation_space)
        action_sizes = self.extract_sizes(env.action_space)
        self.env = env
        self.agents = env.agents
        self.step_count = 0
        self.episode_count = 0
        self.reward = [0] * n_agents

        self.render_frames = []

        #Tuple to Array
        observation_space = np.asarray(list(env.observation_space))
        
        action_space = np.asarray(list(env.action_space))

        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_sizes = observation_sizes
        self.action_sizes = action_sizes

        self.num_actions = int(self.action_space[0].n)
        self.observation_dim = int(np.prod(self.observation_space[0].shape))
        self.dim_actions = 1
        self.dim = 7
        self.true = self.ground_truth()
        self.carryShelf = [-1] * n_agents


    def _action_mask(self):
        # The rest of the codebase unpacks an action mask from reset()/step()
        # (see trainer.get_episode), but never reads it. RWARE has no invalid
        # actions, so hand back an all-ones mask of the right shape.
        return np.ones((len(self.agents), self.num_actions), dtype=int)

    def reset(self):
        reset = np.asarray(list(self.env.reset()))
        #reset = self.getObs(reset)
        reset = reset[np.newaxis]
        reset = torch.tensor(reset, dtype=torch.float64)
        self.true = self.ground_truth()
        self.carryShelf = [-1, -1]

        return reset, self._action_mask()

    def step(self, action):
        # translate_action() hands back one entry per action head (the env
        # action, plus the comm action when hard_attn is on). GymWrapper drops
        # the extra heads for single-action envs; do the same here since this
        # env is used unwrapped.
        if self.dim_actions == 1:
            action = action[0]
        action = tuple(action)
        next_obs, reward, done, info = self.env.step(action)

        #print(len(next_obs[0]), next_obs)
        #next_obs = self.getObs(next_obs)
        if done[0] == False:
            done = False
        else:
            done = True

        # Save the video
        # is_save_video = False
        # if is_save_video:
        #     self.render_frames.append(self.env.render(mode='rgb_array'))
            
        #     for i in range(len(self.reward)):
        #         self.reward[i] += reward[i]
        # if done:
        #     if is_save_video:
        #         save_video(
        #             self.render_frames,
        #             "videos",
        #             fps=10,
        #             step_starting_index=self.step_count,
        #             episode_index=self.episode_count,
        #         )
        #         self.step_count += 1
        #         self.episode_count += 1

        #         print("Video saved successfully!")
        #         print("Agent Reward: ", self.reward)

        # Same (1, nagents, obs_dim) shape GymWrapper._flatten_obs produces, and
        # the same shape reset() returns above -- comm.forward_state_encoder
        # does not squeeze its input.
        next_obs = np.asarray(list(next_obs)).reshape(1, -1, self.observation_dim)
        next_obs = torch.tensor(next_obs, dtype=torch.float64)

        self.true = self.ground_truth()
        self.agents = self.env.agents
        # self.carryShelf = self.target()
        return next_obs, self._action_mask(), np.asarray(reward), done, info

    def render(self):
        self.env.render()
        time.sleep(0.1)

    # trainer/main call display()/end_display() on the env (see GymWrapper).
    def display(self):
        self.render()

    def end_display(self):
        pass


    # Original 3 layer ground truth
    # def ground_truth(self):
    #     env = self.env

    #     arr = np.copy(env.grid[0][0])  # Copy the first layer of the grid (agents)
    #     agents = env.grid[0]
    #     for row in agents:
    #         line = []
    #         for column in row:
    #             # If isn't 0
    #             if column != 0:
    #                 agentNum = int(column)
    #                 agentDirection = env.agents[agentNum-1].dir.value + 1
    #                 line.append(agentDirection)
    #             else:
    #                 line.append(0)
    #         temp = np.array(line)
    #         arr = np.vstack((arr, temp))
    #     arr = arr[1:]

    #     arr2 = np.copy(env.grid[1][0])
    #     boxes = env.grid[1]
    #     for row in boxes:
    #         line = []
    #         for column in row:
    #             # if isn't 0
    #             if column != 0:
    #                 line.append(1)
    #             else:
    #                 line.append(0)
    #         temp = np.array(line)
    #         arr2 = np.vstack((arr2, temp))
        
    #     arr2 = arr2[1:]

    #     arr3 = np.zeros([len(env.grid[1]), len(env.grid[1][0])], dtype=int)
    #     for request in env.request_queue:
    #         x = request.x
    #         y = request.y
    #         arr3[y][x] = 1
        
    #     total = np.stack((arr, arr2, arr3), axis=0)
    #     total = total.flatten()

    #     return total
    
    # Ground truth node feature matrix
    # def ground_truth(self):
    #     env = self.env
    #     num_identities = 2

    #     total_feature_matrix = []

    #     for agent in env.agents:
    #         # Normalize agent location between 0 and 1
    #         agent_location_x = agent.x
    #         agent_location_x = agent_location_x / (len(env.grid[0][0]) - 1)
    #         agent_location_y = agent.y
    #         agent_location_y = agent_location_y / (len(env.grid[0]) - 1)

    #         if agent_location_x > 1:
    #             raise ValueError("Agent location x is out of bounds: {}".format(agent_location_x))
    #         if agent_location_y > 1:
    #             raise ValueError("Agent location y is out of bounds: {}".format(agent_location_y))

    #         # Agent Direction One-hot encoding
    #         agent_direction = agent.dir.value
    #         agent_direction_one_hot = [0] * 4
    #         agent_direction_one_hot[agent_direction] = 1



    #         one_hot_identity = [0] * num_identities
            
    #         # Agent Identity
    #         one_hot_identity[0] = 1

    #         total_feature_matrix.append([agent_location_x, agent_location_y] + agent_direction_one_hot + one_hot_identity)
        
    #     # # Box Feature Matrix
    #     # for box in env.shelfs:
    #     #     box_location_x = box.x
    #     #     box_location_y = box.y

    #     #     one_hot_identity = [0] * num_identities
            
    #     #     # Box Identity
    #     #     one_hot_identity[1] = 1

    #     #     total_feature_matrix.append([box_location_x, box_location_y, 0] + one_hot_identity)

    #     for request in env.request_queue:

    #         # Only add request if agent is in sensor range
    #         agent_in_sensor_range = False
    #         for agent in env.agents:
    #             if (abs(agent.x - request.x) <= 1 and abs(agent.y - request.y) <= 1):
    #                 agent_in_sensor_range = True
    #                 break
            
    #         # If no agent is in sensor range, skip this request
    #         if not agent_in_sensor_range:
    #             total_feature_matrix.append([0] * (2 + 4 + num_identities))
    #             continue

    #         # Normalize request location between 0 and 1
    #         request_location_x = request.x
    #         request_location_x = request_location_x / (len(env.grid[0][0]) - 1)
    #         request_location_y = request.y
    #         request_location_y = request_location_y / (len(env.grid[0]) - 1)

    #         if request_location_x > 1:
    #             raise ValueError("Request location x is out of bounds: {}".format(request_location_x))
    #         if request_location_y > 1:
    #             raise ValueError("Request location y is out of bounds: {}".format(request_location_y))

    #         one_hot_identity = [0] * num_identities
            
    #         # Request Identity
    #         one_hot_identity[1] = 1

    #         total_feature_matrix.append([request_location_x, request_location_y] + ([0] * 4) + one_hot_identity)

    #     total_feature_matrix = np.array(total_feature_matrix)
    #     total_feature_matrix = total_feature_matrix.flatten()

    #     return total_feature_matrix

    # Unnormalized ground truth node feature matrix
    def ground_truth(self):
        env = self.env
        num_identities = 3

        total_feature_matrix = []

        for agent in env.agents:
            agent_location_x = agent.x
            agent_location_y = agent.y

            agent_direction = agent.dir.value+1

            one_hot_identity = [0] * num_identities
            
            # Agent Identity
            one_hot_identity[0] = 1

            total_feature_matrix.append([agent_location_x, agent_location_y, agent_direction] + one_hot_identity)
        
        # for box in env.shelfs:
        #     box_location_x = box.x
        #     box_location_y = box.y

        #     one_hot_identity = [0] * num_identities
            
        #     # Box Identity
        #     one_hot_identity[1] = 1

        #     total_feature_matrix.append([box_location_x, box_location_y, 0] + one_hot_identity)

        for request in env.request_queue:
            request_location_x = request.x
            request_location_y = request.y

            one_hot_identity = [0] * num_identities
            
            # Request Identity
            one_hot_identity[2] = 1

            total_feature_matrix.append([request_location_x, request_location_y, 0] + one_hot_identity)

        total_feature_matrix = np.array(total_feature_matrix)
        total_feature_matrix = total_feature_matrix.flatten()

        return total_feature_matrix

    def target(self):
        env = self.env
        carry = self.carryShelf

        agents = env.agents

        count = 0
        for agent in agents:
            agent_location_x = agent.x
            agent_location_y = agent.y

            found = False
            if agent.carrying_shelf is not None:    
                for request in env.request_queue:
                    x = request.x
                    y = request.y

                    if agent_location_x == x and agent_location_y == y:
                        carry[count] += 1
                        found = True
            if not found:
                carry[count] = -1

            count += 1

        return carry

            

    def extract_sizes(self, spaces):
        """
        Extract space dimensions
        :param spaces: list of Gym spaces
        :return: list of ints with sizes for each agent
        """
        sizes = []
        for space in spaces:
            if isinstance(space, Box):
                size = sum(space.shape)
            elif isinstance(space, Dict):
                size = sum(self.extract_sizes(space.values()))
            elif isinstance(space, Discrete) or isinstance(space, MultiBinary):
                size = space.n
            elif isinstance(space, MultiDiscrete):
                size = sum(space.nvec)
            else:
                raise ValueError("Unknown class of space: ", type(space))
            sizes.append(size)
        return sizes