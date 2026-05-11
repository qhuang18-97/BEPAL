# pp_hallu_scan_all.py
import argparse
import os
import sys
from inspect import getargspec

import numpy as np
import torch
import torch.nn.functional as F

import data
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from action_utils import select_action, translate_action

torch.set_default_tensor_type('torch.DoubleTensor')


parser = argparse.ArgumentParser(description='BEPAL PP hallucination scan (all agents, all steps)')

# ===== 基本训练 / 环境参数（和 main.py 对齐） =====
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--epoch_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--nprocesses', type=int, default=1)

parser.add_argument('--hid_size', default=64, type=int)
parser.add_argument('--recurrent', action='store_true', default=False)

parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--normalize_rewards', action='store_true', default=False)
parser.add_argument('--lrate', type=float, default=0.001)
parser.add_argument('--entr', type=float, default=0)
parser.add_argument('--value_coeff', type=float, default=0.01)

parser.add_argument('--env_name', default="predator_prey")
parser.add_argument('--max_steps', default=40, type=int)
parser.add_argument('--nactions', default='1', type=str)
parser.add_argument('--action_scale', default=1.0, type=float)
parser.add_argument('--obstacles', default=10, type=int)

parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--plot_env', default='main', type=str)
parser.add_argument('--save', default='', type=str)
parser.add_argument('--save_every', default=0, type=int)
parser.add_argument('--load', default='', type=str, help='load the trained model')
parser.add_argument('--display', action="store_true", default=False)
parser.add_argument('--random', action='store_true', default=False)

# CommNet / IC3Net
parser.add_argument('--commnet', action='store_true', default=False)
parser.add_argument('--ic3net', action='store_true', default=False)
parser.add_argument('--nagents', type=int, default=5)
parser.add_argument('--comm_mode', type=str, default='avg')
parser.add_argument('--comm_passes', type=int, default=1)
parser.add_argument('--comm_mask_zero', action='store_true', default=False)
parser.add_argument('--mean_ratio', default=1.0, type=float)
parser.add_argument('--rnn_type', default='MLP', type=str)
parser.add_argument('--detach_gap', default=10, type=int)
parser.add_argument('--comm_init', default='uniform', type=str)
parser.add_argument('--hard_attn', default=False, action='store_true')
parser.add_argument('--comm_action_one', default=False, action='store_true')
parser.add_argument('--advantages_per_action', default=False, action='store_true')
parser.add_argument('--share_weights', default=False, action='store_true')

# 幻觉优化参数
parser.add_argument('--latent_lr', type=float, default=0.05)
parser.add_argument('--latent_steps', type=int, default=100)

# 日志目录
parser.add_argument('--hallu_log_dir', type=str, default='hallucination_logs',
                    help='directory to save per-agent hallucination logs')


def load_model_and_env():
    """和你之前版本类似：构建 env + policy_net，并载入 checkpoint。"""
    init_args_for_env(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])

    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0
        if args.env_name == "traffic_junction":
            args.comm_action_one = True

    args.nfriendly = args.nagents
    if hasattr(args, 'enemy_comm') and getattr(args, 'enemy_comm', False):
        if hasattr(args, 'nenemies'):
            args.nagents += args.nenemies
        else:
            raise RuntimeError("Env. needs to pass argument 'nenemy'.")

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

    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'

    parse_action_args(args)

    if args.seed == -1:
        args.seed = np.random.randint(0, 10000)
    torch.manual_seed(args.seed)

    print(args, flush=True)

    policy_net = CommNetMLP(args, num_inputs)
    policy_net.eval()
    for p in policy_net.parameters():
        p.requires_grad_(False)

    if args.load != '':
        current_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        d = torch.load(current_path + args.load, map_location='cpu')
        policy_net.load_state_dict(d['policy_net'])
        print(f"[HallucinationScan] loaded model from {current_path + args.load}")

    return args, env, policy_net


def optimize_hidden_for_agent(args, policy_net, hidden_state_full, agent_idx):
    """
    对单个 agent 的 hidden 做一次 prey 幻觉：
      - target prey 坐标：对角线翻转 (1 - x, 1 - y)
      - 只动这个 agent 的 hidden，其它 agent 不变
      - 返回前后 belief + action 的对比信息
    """
    nagents = args.nagents
    hid_size = args.hid_size
    prey_idx = nagents

    h_all = hidden_state_full.view(nagents, hid_size)  # [N, hid]
    h_i_orig = h_all[agent_idx].detach()
    h_i_opt = h_i_orig.clone().detach().requires_grad_(True)

    # 原始 belief
    with torch.no_grad():
        pred_orig = policy_net.mapdecode(h_i_orig.unsqueeze(0))  # [1,(N+1)*4]
    pred_orig_slot = pred_orig.view(1, nagents + 1, 4)          # [1,N+1,4]

    x_orig = pred_orig_slot[0, prey_idx, 0].item()
    y_orig = pred_orig_slot[0, prey_idx, 1].item()

    # 对角线幻觉：1 - x, 1 - y
    target_x = 1.0 - x_orig
    target_y = 1.0 - y_orig
    target_x = max(0.0, min(1.0, target_x))
    target_y = max(0.0, min(1.0, target_y))

    target_slot = pred_orig_slot.clone().detach()
    target_slot[0, prey_idx, 0] = target_x
    target_slot[0, prey_idx, 1] = target_y

    optimizer = torch.optim.Adam([h_i_opt], lr=args.latent_lr)

    for step in range(args.latent_steps):
        optimizer.zero_grad()
        pred_cur = policy_net.mapdecode(h_i_opt.unsqueeze(0))
        pred_cur_slot = pred_cur.view(1, nagents + 1, 4)

        prey_cur = pred_cur_slot[:, prey_idx, :2]
        prey_tgt = target_slot[:, prey_idx, :2]

        others_cur = pred_cur_slot[:, :prey_idx, :]
        others_tgt = pred_orig_slot[:, :prey_idx, :]

        loss_target = F.mse_loss(prey_cur, prey_tgt)
        loss_stable = F.mse_loss(others_cur, others_tgt)
        loss_reg = F.mse_loss(h_i_opt, h_i_orig)

        loss = loss_target + 0.1 * loss_stable + 0.01 * loss_reg
        loss.backward()
        optimizer.step()

    h_all_new = h_all.clone()
    h_all_new[agent_idx] = h_i_opt.detach()

    # 幻觉后的 belief
    with torch.no_grad():
        pred_new = policy_net.mapdecode(h_all_new[agent_idx].unsqueeze(0)).view(
            1, nagents + 1, 4
        )

    # 动作分布对比
    h_orig_batch = h_all.view(1, nagents, hid_size)
    h_new_batch = h_all_new.view(1, nagents, hid_size)

    with torch.no_grad():
        logits_orig = [head(h_orig_batch) for head in policy_net.heads]
        logits_new = [head(h_new_batch) for head in policy_net.heads]

        logit_o = logits_orig[0][0, agent_idx, :]
        logit_n = logits_new[0][0, agent_idx, :]

        prob_o = F.softmax(logit_o, dim=-1)
        prob_n = F.softmax(logit_n, dim=-1)

    delta_norm = torch.norm(h_all_new[agent_idx] - h_all[agent_idx]).item()

    result = {
        'prey_orig': (x_orig, y_orig),
        'prey_target': (target_x, target_y),
        'prey_new': (
            pred_new[0, prey_idx, 0].item(),
            pred_new[0, prey_idx, 1].item()
        ),
        'delta_h': delta_norm,
        'logits_orig': logit_o.detach().cpu().numpy(),
        'logits_new': logit_n.detach().cpu().numpy(),
        'probs_orig': prob_o.detach().cpu().numpy(),
        'probs_new': prob_n.detach().cpu().numpy(),
        'argmax_orig': int(torch.argmax(prob_o).item()),
        'argmax_new': int(torch.argmax(prob_n).item()),
    }
    return result


def main():
    args, env, policy_net = load_model_and_env()

    # 准备 log 目录 & 每个 agent 各自一个文件
    os.makedirs(args.hallu_log_dir, exist_ok=True)
    log_files = []
    for i in range(args.nagents):
        path = os.path.join(args.hallu_log_dir, f'agent_{i}.log')
        f = open(path, 'w', encoding='utf-8')
        f.write(f"# Hallucination log for agent {i}\n")
        f.write(f"# env={args.env_name}, load={args.load}\n\n")
        log_files.append(f)

    # ===== 只跑 1 个 episode =====
    reset_args = getargspec(env.reset).args
    if 'epoch' in reset_args:
        state, action_mask = env.reset(0)
    else:
        state, action_mask = env.reset()

    info = {}
    nagents = args.nagents

    if args.recurrent:
        prev_hid = policy_net.init_hidden(batch_size=state.shape[0])
    else:
        prev_hid = None

    for t in range(args.max_steps):
        if args.hard_attn and args.commnet and t == 0:
            info['comm_action'] = np.zeros(nagents, dtype=int)

        if args.recurrent:
            x = [state, prev_hid]
            with torch.no_grad():
                action_out, value, value_global, prev_hid, node_decoded = policy_net(x, info)
        else:
            raise RuntimeError("当前脚本假设使用 recurrent=LSTM / IC3Net")

        hidden_state_full = prev_hid[0].detach().clone()

        # 当前真实坐标（方便 log）
        prey_pos = env.env.prey_loc[0]          # [row, col]
        agent_pos_all = env.env.predator_loc    # list of [row, col]

        # 对每个 agent 做一次 prey 幻觉 + log
        for agent_idx in range(nagents):
            res = optimize_hidden_for_agent(args, policy_net, hidden_state_full, agent_idx)

            f = log_files[agent_idx]
            f.write(f"=== step {t} ===\n")
            f.write(f"real_agent_pos (row,col): {agent_pos_all[agent_idx]}\n")
            f.write(f"real_prey_pos  (row,col): {prey_pos}\n")
            # ===== 新增：周围障碍物信息 =====
            obs_grid = env.env.obstacle_grid  # [M, M]，0/1 网格
            M = obs_grid.shape[0]
            r, c = agent_pos_all[agent_idx]

            def is_blocked(rr, cc):
                # 越界也视为“挡住”（相当于墙）
                if rr < 0 or rr >= M or cc < 0 or cc >= M:
                    return True
                return obs_grid[rr, cc] == 1

            up_blocked = is_blocked(r - 1, c)
            right_blocked = is_blocked(r, c + 1)
            down_blocked = is_blocked(r + 1, c)
            left_blocked = is_blocked(r, c - 1)

            f.write(
                "neighbors_blocked (U,R,D,L): {}, {}, {}, {}\n".format(
                    int(up_blocked), int(right_blocked),
                    int(down_blocked), int(left_blocked)
                )
            )

            # 再打印一个以 agent 为中心的局部障碍 patch（用 vision 半径）
            vis = getattr(args, "vision", 2)
            r0, r1 = max(0, r - vis), min(M, r + vis + 1)
            c0, c1 = max(0, c - vis), min(M, c + vis + 1)
            local_patch = obs_grid[r0:r1, c0:c1]

            f.write(
                f"local_obstacle_patch rows[{r0}:{r1}), cols[{c0}:{c1}):\n"
            )
            f.write(str(local_patch) + "\n")
            f.write("prey_belief_orig   (x,y): {:.6f}, {:.6f}\n".format(
                res['prey_orig'][0], res['prey_orig'][1]))
            f.write("prey_belief_target (x,y): {:.6f}, {:.6f}\n".format(
                res['prey_target'][0], res['prey_target'][1]))
            f.write("prey_belief_new    (x,y): {:.6f}, {:.6f}\n".format(
                res['prey_new'][0], res['prey_new'][1]))

            f.write("delta_h (L2 norm): {:.6f}\n".format(res['delta_h']))
            f.write("logits_orig: {}\n".format(res['logits_orig']))
            f.write("logits_new : {}\n".format(res['logits_new']))
            f.write("probs_orig : {}\n".format(res['probs_orig']))
            f.write("probs_new  : {}\n".format(res['probs_new']))
            f.write("argmax_orig: {}\n".format(res['argmax_orig']))
            f.write("argmax_new : {}\n".format(res['argmax_new']))
            f.write("\n")

        # ===== env 前进一步（用原始 action_out）=====
        action = select_action(args, action_out)
        action, actual = translate_action(args, env, action)
        next_state, action_mask, reward, done, info = env.step(actual)

        if args.hard_attn and args.commnet:
            if not args.comm_action_one:
                info['comm_action'] = action[-1]
            else:
                info['comm_action'] = np.ones(nagents, dtype=int)

        state = next_state
        if done:
            print(f"[HallucinationScan] episode ended at step {t}")
            break

    for f in log_files:
        f.close()
    print(f"[HallucinationScan] logs saved to directory: {args.hallu_log_dir}")


if __name__ == "__main__":
    main()
