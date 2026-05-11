import argparse
import os
import sys
import numpy as np
import torch
import data
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args, select_action, translate_action
import inspect
from inspect import getargspec

torch.set_default_tensor_type('torch.DoubleTensor')


def get_ground_truth_from_env(env, args):
    """
    返回:
    1. gt_tensor: [N_entities, 4] (Agents + Prey)
    2. obstacles: [N_obs, 2] (障碍物坐标)
    3. all_on_prey: bool (是否所有agent都抓到了prey)
    """
    real_env = env
    while hasattr(real_env, 'env'):
        real_env = real_env.env

    grid_dim = getattr(args, 'dim', 12)
    scale = grid_dim - 1

    # --- 1. 获取 Agents & Prey ---
    gt_tensor = torch.zeros(args.nagents + 1, 4)
    all_on_prey = False

    if hasattr(real_env, 'predator_loc') and hasattr(real_env, 'prey_loc'):
        # 获取原始坐标用于判定距离
        pred_loc = real_env.predator_loc
        prey_loc = real_env.prey_loc

        # 判定是否所有 Agent 都抓到了 Prey (距离极小)
        # 注意：这里使用未归一化的坐标计算距离更直观，或者归一化后用很小的阈值
        dists = np.linalg.norm(pred_loc - prey_loc, axis=1)
        # 判定阈值设为 0.05 (稍微宽容一点点，防止浮点误差，实际重叠通常是 0)
        # 如果是 Grid World 整数坐标，重叠就是 0
        all_on_prey = np.all(dists < 0.05)

        locs = np.concatenate((pred_loc, prey_loc), axis=0)
        norm_locs = locs / scale
        gt_tensor[:, :2] = torch.from_numpy(norm_locs)

    # --- 2. 获取障碍物 ---
    obs_list = []
    if hasattr(real_env, 'obstacle_grid'):
        obs_coords = np.argwhere(real_env.obstacle_grid == 1)
        if len(obs_coords) > 0:
            norm_obs = obs_coords / scale
            obs_list = norm_obs

    if len(obs_list) == 0:
        obstacles = torch.zeros(0, 2)
    else:
        obstacles = torch.tensor(obs_list).float()

    return gt_tensor.float(), obstacles, all_on_prey


def load_model_and_env(args_list=None):
    parser = argparse.ArgumentParser(description='Data Collection for BEPAL XAI')
    # ... (参数保持不变) ...
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
    parser.add_argument('--save_path', type=str, default='Probe_case/data_h_with_gt_clean.pt')

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
            print(f"[DataCollect] Loaded model from {model_path}")
        else:
            print(f"[Warning] Model path not found: {model_path}")

    return args, env, policy_net


def collect_data():
    args, env, policy_net = load_model_and_env()
    collected_data = []

    print(f"Start collecting {args.num_episodes} episodes (Stopping at capture)...")

    for ep in range(args.num_episodes):
        reset_args = inspect.getfullargspec(env.reset).args
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

            network_action = select_action(args, action_out)

            # ================= [修复核心] =================
            # 目标：无论 select_action 返回什么妖魔鬼怪，必须提取出 [Batch, Nagents] 或者 [Nagents] 的物理动作
            physical_action = network_action

            # Case 1: 它是 List 或 Tuple (如 [act, comm])
            if isinstance(network_action, (list, tuple)):
                physical_action = network_action[0]

            # Case 2: 它是 Tensor，且第一个维度是 2 (对应 Phy/Comm)，且 Nagents != 2 (防止歧义)
            # 你的截图显示形状可能是 (2, 1, 5, 1)，这意味着 dim 0 是 Split
            elif torch.is_tensor(network_action) and network_action.shape[0] == 2:
                # 只有当 dim 0 是 2，而我们需要的 nagents 是 5 时，才敢断定这是 [Phy, Comm] 堆叠
                if args.nagents > 2:
                    physical_action = network_action[0]
            # ============================================

            action_trans, actual = translate_action(args, env, network_action)

            if args.recurrent:
                current_h = next_hid[0].detach().cpu().squeeze(0)
                current_gt, current_obs, all_on_prey = get_ground_truth_from_env(env, args)

                # 转 Tensor
                if isinstance(physical_action, list):
                    action_to_save = torch.tensor(physical_action)
                elif isinstance(physical_action, torch.Tensor):
                    action_to_save = physical_action.cpu()
                else:
                    action_to_save = torch.tensor(physical_action)

                # 暴力降维：直到它变成一维向量 [Nagents]
                # 你的截图显示它是 (2, 1, 5, 1) -> 取 [0] 后变成 (1, 5, 1) -> squeeze 后变成 (5,)
                while action_to_save.dim() > 1:
                    action_to_save = action_to_save.squeeze()

                # 【终极保险】如果 squeeze 过头变成了 scalar (比如 nagents=1)，或者形状不对
                if action_to_save.numel() == args.nagents:
                    action_to_save = action_to_save.view(args.nagents)
                else:
                    # 如果这行被触发，说明前面的拆包逻辑漏了某些 case，但在你的设置下应该不会
                    print(f"Warning: Unexpected action shape {action_to_save.shape} for nagents={args.nagents}")

                for ag_id in range(args.nagents):
                    # 防止 ag_id 越界 (虽然理论上现在 shape 已经对齐了)
                    if ag_id < action_to_save.shape[0]:
                        act_val = action_to_save[ag_id].clone()
                    else:
                        act_val = torch.tensor(0)  # Fallback

                    item = {
                        'h': current_h[ag_id].clone(),
                        'gt': current_gt.clone(),
                        'obs': current_obs.clone(),
                        'action': act_val,
                        'step': t,
                        'episode': ep,
                        'agent_id': ag_id
                    }
                    collected_data.append(item)

                if all_on_prey:
                    break

            next_state, action_mask, reward, done, info = env.step(actual)

            state = next_state
            prev_hid = next_hid

            if args.hard_attn and args.commnet:
                # 同样处理 Comm Action 的提取
                comm_act_source = network_action

                if isinstance(network_action, (list, tuple)):
                    comm_act_source = network_action[1]
                elif torch.is_tensor(network_action) and network_action.shape[0] == 2 and args.nagents > 2:
                    comm_act_source = network_action[1]
                # 如果是其他情况，尝试 fallback 到 -1 (不一定对，但在你的case里应该走上面两个分支)
                elif isinstance(network_action, (list, tuple)) or (
                        torch.is_tensor(network_action) and network_action.dim() > 1):
                    # 这是一个无奈的 fallback
                    pass

                if not args.comm_action_one:
                    comm_act = comm_act_source
                    # 各种处理把它变 numpy
                    if isinstance(comm_act, torch.Tensor):
                        comm_act = comm_act.cpu().numpy()
                    if len(comm_act.shape) > 1:
                        comm_act = comm_act.flatten()  # 简单粗暴展平
                        # 只要取前 nagents 个即可
                        if comm_act.size >= args.nagents:
                            comm_act = comm_act[:args.nagents]

                    info['comm_action'] = comm_act
                else:
                    info['comm_action'] = np.ones(args.nagents, dtype=int)

            if done: break

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{args.num_episodes} finished.")

    print("Collating data...")
    final_dataset = {
        'h': torch.stack([x['h'] for x in collected_data]),
        'gt': torch.stack([x['gt'] for x in collected_data]),
        'action': torch.stack([x['action'] for x in collected_data]),
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