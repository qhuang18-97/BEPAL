import copy
from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'value_global', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc', 'node_ground_truthg', 'location', 'node_decoded', 'move_action'))# v5 added move_action

'''
Dyanamic Graph version trainer
6.5 update: v5: predict 0/1 action rather than vector
'''



class Trainer(object):
    def __init__(self, args, policy_net,  env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        
    
    def ground_truth_gen(self, env):
        """
        Generate ground truth of shape [N, F] for Traffic Junction.
        Each row i contains the features of agent i:
        [normalized location, ID one-hot, route embedding]

        Returns:
            - node_ground_truth: [N, F] for all agents (alive or not)
            - locs_norm: [N, 2] normalized positions (optional)
        """
        car_loc = env.car_loc  # [N, 2]
        dims = env.dims[0]
        n_agents = self.args.nagents
        # route_ids = torch.tensor(env.route_id, dtype=torch.long)  # [N]
        # route_ids[route_ids == -1] =  env.npath

        # Normalize location
        locs_norm = torch.tensor(car_loc / (dims - 1), dtype=torch.float32)  # [N, 2]

        # Identity matrix
        id_matrix = torch.eye(n_agents)  # [N, N]

        # Route embedding lookup
        # route_emb = self.policy_net.route_embed(route_ids)  # [N, embed_dim]

        # Concatenate full feature: [loc | ID | route_emb]
        node_gt = torch.cat([locs_norm, id_matrix], dim=1)  # [N, F], route_emb

        return node_gt, locs_norm #


    def get_episode(self, epoch):
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        
        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)
            
            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                # change x to graph as
                # node, adj = self.state2graph(self.env.env)
                # x = [node, adj, prev_hid]
                
                x = [state, prev_hid]
                action_out, value, value_global, prev_hid, node_decoded = self.policy_net(x, info) #, grid_decoded

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value ,value_global = self.policy_net(x, info)
            # print("action_out", action_out[0].shape, action_out[0].min(), action_out[0].max())
            # print("action_out", action_out[1].shape, action_out[1].min(), action_out[1].max())
            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            
            node_decoded = node_decoded.view(self.args.nagents, self.args.nagents, -1) # +8+2(self.args.nagents+2)
            
            
            node_gt, location = self.ground_truth_gen(self.env.env)
            
            node_ground_truthg = node_gt.unsqueeze(0).repeat(self.args.nagents, 1, 1)  # [N, N, F]
            
            next_state, reward, done, info = self.env.step(actual)
            move_action = torch.tensor(action[0])
           
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                
                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()


            trans = Transition(state, action, action_out, value, value_global, episode_mask, episode_mini_mask, next_state, reward, misc, node_ground_truthg,  location, node_decoded, move_action)
            # trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc, maploss) grid_maploss,

            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)
        
        values = torch.cat(batch.value, dim=0)
        values_g = torch.cat(batch.value_global, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        #alive_masks = alive_masks.to(self.device)
        coop_returns = torch.Tensor(batch_size, n)#.cuda()
        ncoop_returns = torch.Tensor(batch_size, n)#.cuda()
        returns = torch.Tensor(batch_size, n)#.cuda()
        deltas = torch.Tensor(batch_size, n)#.cuda()
        advantages = torch.Tensor(batch_size, n)#.cuda()
        values = values.view(batch_size, n)
        values_g = values_g.view(batch_size, n,n) # 

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        node_decoded = torch.stack(batch.node_decoded, dim=0)
        action_decoded = torch.stack(batch.move_action, dim=0)
        node_loc = torch.stack(batch.node_ground_truthg, dim=0)
        
        location = torch.stack(batch.location, dim=0)
        # vector = torch.zeros((rewards.size(0), 1, rewards.size(1), 2))  # [T, 1, N, 2]

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]
            # scale_factor = 1  # or dim-1 if you want to normalize
            # if i >= rewards.size(0) - 1:
            #     vector[i][0] = location[-1] / scale_factor - location[i] / scale_factor
            # else:
            #     contains_zero_row = torch.all(episode_masks[i:i + 1] == episode_masks[-1], dim=1)
            #     if contains_zero_row.any():
            #         index = contains_zero_row.nonzero(as_tuple=True)[0][0].item()
            #         vector[i][0] = location[i + index] / scale_factor - location[i] / scale_factor
            #     else:
            #         vector[i][0] = location[i + 1] / scale_factor - location[i] / scale_factor

        # vector = vector.repeat(1, n, 1, 1)  # [T, N, N, 2]  v3
        # node_gt = node_loc # torch.cat((node_loc, vector ), dim=3) v3  #[T, N, N, 2 + ...] v4
        action_decoded = action_decoded.view(rewards.size(0), 1, rewards.size(1), 1).repeat(1, n, 1, 1)  # [T, N, N, 2] v5
        node_gt = torch.cat((node_loc, action_decoded), dim=3) # v5
        # node_gt = node_loc
        Loss_func = nn.MSELoss(reduction='none') #sum
        node_maploss = Loss_func(node_decoded, node_gt.detach())#.sum(dim=[1,2, 3]) /((n+1)*2)
        # node_alive_masks = alive_masks.view(node_maploss.shape[0], node_maploss.shape[2])
        # mask = node_alive_masks.unsqueeze(1).unsqueeze(-1)  # [T, 1, N, 1]
        # masked_loss = node_maploss * mask
        loss_per_target = node_maploss.sum(dim=[-1,-2]).view(-1)
        
        map_loss = loss_per_target*alive_masks
        

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)


        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks


        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()
        # gloable value loss term
        #targets_g = returns.sum(1).view(batch_size,1)
        targets_g = returns.unsqueeze(1).repeat(1, n, 1)
        # # value_loss_g = (values_g/self.args.nagents - targets_g/self.args.nagents).pow(2).view(-1)
        value_loss_g = (values_g - targets_g).pow(2).view(-1) # Feb setting
        value_loss_g *= alive_masks.repeat(n).view(-1)  #
        value_loss_g = value_loss_g.sum()

        stat['value_loss'] = value_loss.item()
        stat['value_loss_g'] = (value_loss_g/self.args.nagents).item() #

        map_loss = (map_loss).sum()  # masked_loss
        stat['map_loss'] = map_loss.item()
        loss = action_loss + self.args.value_coeff * (value_loss)  + map_loss/(self.args.nagents*(self.args.nagents+2)) #+ self.args.value_coeff/self.args.nagents * (value_loss_g)#   0.5* 


        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy


        stat['loss'] = loss.item()
        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size: # commended for data collection
        # while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
