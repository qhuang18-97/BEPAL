import argparse
import os
import sys
import numpy as np
import torch
import data
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args, select_action, translate_action
from inspect import getargspec

torch.set_default_tensor_type('torch.DoubleTensor')


# --- 新增：从 Trainer 复刻的 GT 提取函数 ---
def get_ground_truth_from_env(env, args):
    """
    返回:
    1. gt_tensor: [N_entities, 4] (Agents + Prey)
    2. obstacles: [N_obs, 2] (障碍物坐标)
    """
    real_env = env
    while hasattr(real_env, 'env'):
        real_env = real_env.env

    grid_dim = getattr(args, 'dim', 12)
    scale = grid_dim - 1

    # --- 1. 获取 Agents & Prey (保持不变) ---
    gt_tensor = torch.zeros(args.nagents + 1, 4)
    if hasattr(real_env, 'predator_loc') and hasattr(real_env, 'prey_loc'):
        locs = np.concatenate((real_env.predator_loc, real_env.prey_loc), axis=0)
        norm_locs = locs / scale
        gt_tensor[:, :2] = torch.from_numpy(norm_locs)

    # --- 2. 新增：获取障碍物 ---
    obs_list = []
    # 检查是否有 obstacle_grid (通常是 numpy array)
    if hasattr(real_env, 'obstacle_grid'):
        # 找到所有值为 1 (或非0) 的坐标
        # np.argwhere 返回的是 [[r, c], [r, c]...]
        obs_coords = np.argwhere(real_env.obstacle_grid == 1)  # 或者 > 0
        if len(obs_coords) > 0:
            # 注意：有些环境 grid 是 [x, y]，有些是 [row, col] (即 y, x)
            # IC3Net通常是 [x][y]，但为了保险，我们假设它是 [x, y]
            norm_obs = obs_coords / scale
            obs_list = norm_obs

    if len(obs_list) == 0:
        obstacles = torch.zeros(0, 2)
    else:
        obstacles = torch.tensor(obs_list).float()

    return gt_tensor.float(), obstacles


def load_model_and_env(args_list=None):
    # ... (保持原有的加载逻辑不变) ...
    parser = argparse.ArgumentParser(description='Data Collection for BEPAL XAI')
    parser.add_argument('--env_name', default="predator_prey", help='predator_prey | traffic_junction | starcraft')
    parser.add_argument('--nagents', type=int, default=5)
    parser.add_argument('--max_steps', default=40, type=int)
    parser.add_argument('--hid_size', default=64, type=int)
    parser.add_argument('--load', default='', type=str, help='model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--recurrent', action='store_true', default=True)
    parser.add_argument('--rnn_type', default='LSTM', type=str)
    parser.add_argument('--nprocesses', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--normalize_rewards', action='store_true', default=False)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--entr', type=float, default=0)
    parser.add_argument('--value_coeff', type=float, default=0.01)
    parser.add_argument('--nactions', default='1', type=str)
    parser.add_argument('--action_scale', default=1.0, type=float)
    parser.add_argument('--obstacles', default=10, type=int)
    parser.add_argument('--commnet', action='store_true', default=False)
    parser.add_argument('--ic3net', action='store_true', default=False)
    parser.add_argument('--comm_mode', type=str, default='avg')
    parser.add_argument('--comm_passes', type=int, default=1)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--mean_ratio', default=1.0, type=float)
    parser.add_argument('--detach_gap', default=10, type=int)
    parser.add_argument('--comm_init', default='uniform', type=str)
    parser.add_argument('--hard_attn', default=False, action='store_true')
    parser.add_argument('--comm_action_one', default=False, action='store_true')
    parser.add_argument('--advantages_per_action', default=False, action='store_true')
    parser.add_argument('--share_weights', default=False, action='store_true')
    parser.add_argument('--display', action="store_true", default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='Probe_case/data_h_with_gt.pt')  # 默认路径改一下方便点

    init_args_for_env(parser)
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args, _ = parser.parse_known_args()

    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0
        if args.env_name == "traffic_junction":
            args.comm_action_one = True

    args.nfriendly = args.nagents
    env = data.init(args.env_name, args, False)

    num_inputs = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.num_inputs = num_inputs
    if args.hard_attn and args.commnet:
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1
    parse_action_args(args)

    policy_net = CommNetMLP(args, num_inputs)
    policy_net.eval()

    if args.load != '':
        current_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        model_path = os.path.join(current_path, args.load)
        if os.path.exists(model_path):
            d = torch.load(model_path, map_location='cpu')
            policy_net.load_state_dict(d['policy_net'])
            print(f"[CommHallucination] loaded model from {model_path}")
        else:
            print(f"[Warning] Model path not found: {model_path}")

    return args, env, policy_net


def collect_data():
    args, env, policy_net = load_model_and_env()
    collected_data = []

    print(f"Start collecting {args.num_episodes} episodes...")

    for ep in range(args.num_episodes):
        reset_args = getargspec(env.reset).args
        if 'epoch' in reset_args:
            state, action_mask = env.reset(ep)
        else:
            state, action_mask = env.reset()

        if args.recurrent:
            prev_hid = policy_net.init_hidden(batch_size=state.shape[0])
        else:
            prev_hid = None

        info = {}

        for t in range(args.max_steps):
            if args.hard_attn and args.commnet and t == 0:
                info['comm_action'] = np.zeros(args.nagents, dtype=int)

            with torch.no_grad():
                action_out, value, value_global, next_hid, node_decoded = policy_net([state, prev_hid], info)

            # --- 保存数据 (修复版) ---
            if args.recurrent:
                current_h = next_hid[0].detach().cpu().squeeze(0)

                # 【修改】同时获取 GT 和 Obstacles
                current_gt, current_obs = get_ground_truth_from_env(env, args)

                for ag_id in range(args.nagents):
                    item = {
                        'h': current_h[ag_id].clone(),
                        'gt': current_gt.clone(),
                        'obs': current_obs.clone(),  # 【新增】保存障碍物
                        'step': t,
                        'episode': ep,
                        'agent_id': ag_id
                    }
                    collected_data.append(item)

            action = select_action(args, action_out)
            action_trans, actual = translate_action(args, env, action)
            next_state, action_mask, reward, done, info = env.step(actual)

            state = next_state
            prev_hid = next_hid

            if args.hard_attn and args.commnet:
                if not args.comm_action_one:
                    comm_act = action[-1]
                    if len(comm_act.shape) > 1: comm_act = comm_act[0]
                    if isinstance(comm_act, torch.Tensor): comm_act = comm_act.cpu().numpy()
                    info['comm_action'] = comm_act
                else:
                    info['comm_action'] = np.ones(args.nagents, dtype=int)

            if done: break

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{args.num_episodes} finished.")

    # --- 保存 ---
    print("Collating data...")
    final_dataset = {
        'h': torch.stack([x['h'] for x in collected_data]),
        'gt': torch.stack([x['gt'] for x in collected_data]),
        # 'obs': torch.stack(...) 如果障碍物数量变动会报错，这里建议存 list
        'obs': [x['obs'] for x in collected_data],
        'step': torch.tensor([x['step'] for x in collected_data], dtype=torch.long),
        'agent_id': torch.tensor([x['agent_id'] for x in collected_data], dtype=torch.long),
        'episode': torch.tensor([x['episode'] for x in collected_data], dtype=torch.long)
    }

    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(final_dataset, save_path)

    print(f"\nCollection Complete! Samples: {len(collected_data)}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    collect_data()