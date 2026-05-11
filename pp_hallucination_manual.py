# pp_hallucination.py
import argparse
import os
import sys
import math
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


# ---------- 1. 拷一份和 main.py 一致的 parser 基本设置 ----------
parser = argparse.ArgumentParser(description='BEPAL PP hallucination experiment')

# training-style args（为了兼容你原来的命令行）
parser.add_argument('--num_epochs', default=1, type=int)              # 用不上，但保留
parser.add_argument('--epoch_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--nprocesses', type=int, default=1)

# model
parser.add_argument('--hid_size', default=64, type=int)
parser.add_argument('--recurrent', action='store_true', default=False)

# optimization (基本不用，但保持一致)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--normalize_rewards', action='store_true', default=False)
parser.add_argument('--lrate', type=float, default=0.001)
parser.add_argument('--entr', type=float, default=0)
parser.add_argument('--value_coeff', type=float, default=0.01)

# environment
parser.add_argument('--env_name', default="predator_prey")
parser.add_argument('--max_steps', default=40, type=int)
parser.add_argument('--nactions', default='1', type=str)
parser.add_argument('--action_scale', default=1.0, type=float)
parser.add_argument('--obstacles', default=10, type=int)

# other
parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--plot_env', default='main', type=str)
parser.add_argument('--save', default='', type=str)
parser.add_argument('--save_every', default=0, type=int)
parser.add_argument('--load', default='', type=str, help='load the trained model')
parser.add_argument('--display', action="store_true", default=False)

parser.add_argument('--random', action='store_true', default=False)

# CommNet specific args
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

# 额外：实验控制参数
parser.add_argument('--hallucinate_agent', type=int, default=0,
                    help='which agent to hallucinate')
parser.add_argument('--hallucinate_step', type=int, default=5,
                    help='which step to hallucinate')
parser.add_argument('--hallucinate_x', type=float, default=0.0,
                    help='target prey normalized x')
parser.add_argument('--hallucinate_y', type=float, default=0.0,
                    help='target prey normalized y')
parser.add_argument('--latent_lr', type=float, default=0.05)
parser.add_argument('--latent_steps', type=int, default=100)

def compute_prey_seen(env, args):
    """
    判断在当前 env 状态下，每个 agent 是否能看到 prey。
    这里按照 vision 范围做一个简单判定：
      abs(dx) <= vision 且 abs(dy) <= vision  → 认为“看到猎物”
    """
    preds = np.array(env.env.predator_loc)   # [N, 2]
    prey = np.array(env.env.prey_loc[0])     # [2]
    seen = np.zeros(args.nagents, dtype=bool)

    for i, p in enumerate(preds):
        dx = abs(p[0] - prey[0])
        dy = abs(p[1] - prey[1])
        if dx <= args.vision and dy <= args.vision:
            seen[i] = True
    return seen


def load_model_and_env(args):
    # init env args
    init_args_for_env(parser)
    # 重新 parse 一次（init_args_for_env 会加更多 env 相关的参数）
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])
    args = parsed_args

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    if not isinstance(args.num_actions, (list, tuple)):  # single action case
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

    if args.load != '':
        current_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        d = torch.load(current_path + args.load, map_location='cpu')
        policy_net.load_state_dict(d['policy_net'])
        print(f"[Hallucination] loaded model from {current_path + args.load}")

    return args, env, policy_net


def pick_step_and_collect_hidden(args, env, policy_net):
    """
    跑一条 episode，在某个时间步拿到:
      - state_t
      - hidden_state_t (所有 agent)
      - action_out_t
      - node_decoded_t
      - seen_ever_prev: 从 0 到 t-1 期间是否有 agent 见过 prey
      - seen_ever_now : 从 0 到 t   期间是否有 agent 见过 prey
    """
    # reset env
    reset_args = getargspec(env.reset).args
    if 'epoch' in reset_args:
        state, action_mask = env.reset(0)
    else:
        state, action_mask = env.reset()

    info = {}
    chosen = None

    # 从 episode 开始到当前为止“累计看到情况”
    seen_ever = np.zeros(args.nagents, dtype=bool)

    prev_hid = torch.zeros(1, args.nagents, args.hid_size)

    for t in range(args.max_steps):
        # 当前这一步“瞬时看到情况”
        seen_now = compute_prey_seen(env, args)
        # 累计：到“本步之前”为止的记录
        seen_ever_prev = seen_ever.copy()
        # 更新到“包括本步”的记录
        seen_ever_now = seen_ever | seen_now

        if t == 0 and args.hard_attn and args.commnet:
            info['comm_action'] = np.zeros(args.nagents, dtype=int)

        if args.recurrent:
            if args.rnn_type == 'LSTM' and t == 0:
                prev_hid = policy_net.init_hidden(batch_size=state.shape[0])
            x = [state, prev_hid]
            with torch.no_grad():
                action_out, value, value_global, prev_hid, node_decoded = policy_net(x, info)
        else:
            raise RuntimeError("当前实验脚本假设使用 recurrent=LSTM")

        hidden_state_full = prev_hid[0].detach().clone()
        node_decoded = node_decoded.view(args.nagents,
                                         args.nagents + 1,
                                         -1).detach().clone()

        if t == args.hallucinate_step:
            chosen = {
                't': t,
                'state': state.clone(),
                'info': dict(info),
                'hidden_state_full': hidden_state_full,
                'action_out': [a.clone() for a in action_out],
                'node_decoded': node_decoded.clone(),
                'seen_ever_prev': seen_ever_prev.copy(),
                'seen_ever_now': seen_ever_now.copy(),
            }
            break

        # 环境往前滚一步
        action = select_action(args, action_out)
        action, actual = translate_action(args, env, action)
        next_state, action_mask, reward, done, info = env.step(actual)

        if args.hard_attn and args.commnet:
            if not args.comm_action_one:
                info['comm_action'] = action[-1]
            else:
                info['comm_action'] = np.ones(args.nagents, dtype=int)

        state = next_state
        # 更新累计“看过”记录
        seen_ever = seen_ever_now
        if done:
            break

    if chosen is None:
        print("[Hallucination] WARNING: did not hit hallucinate_step, fallback to last step.")
        chosen = {
            't': t,
            'state': state.clone(),
            'info': dict(info),
            'hidden_state_full': hidden_state_full,
            'action_out': [a.clone() for a in action_out],
            'node_decoded': node_decoded.clone(),
            'seen_ever_prev': seen_ever_prev.copy(),
            'seen_ever_now': seen_ever_now.copy(),
        }

    print(f"[Hallucination] picked step t = {chosen['t']}")
    return chosen





def optimize_hidden_for_prey_hallucination(args, policy_net, hidden_state_full, node_decoded, agent_idx):
    """
    对单个 agent 的 hidden state 做 latent 优化：
    目标：让 mapdecode 输出的 prey belief 的 (x,y) 变成指定 hallucinate_x, hallucinate_y。
    同时尽量保持其他实体的 belief 不变 + h 不要偏离太多。
    """
    nagents = args.nagents
    hid_size = args.hid_size
    prey_idx = nagents  # 第 N 个 slot 是 prey

    # 取出这个 agent 的 hidden
    h_all = hidden_state_full.view(nagents, hid_size)  # [N, hid]
    h_i_orig = h_all[agent_idx].detach()
    h_i_opt = h_i_orig.clone().detach().requires_grad_(True)

    # 原始 belief
    with torch.no_grad():
        pred_orig = policy_net.mapdecode(h_i_orig.unsqueeze(0))  # [1, (N+1)*4]
    pred_orig_slot = pred_orig.view(1, nagents + 1, 4)          # [1, N+1, 4]

    # 目标 belief：只改 prey 的 x,y 为指定值
    target_slot = pred_orig_slot.clone().detach()
    target_slot[0, prey_idx, 0] = args.hallucinate_x  # x
    target_slot[0, prey_idx, 1] = args.hallucinate_y  # y

    optimizer = torch.optim.Adam([h_i_opt], lr=args.latent_lr)

    for step in range(args.latent_steps):
        optimizer.zero_grad()

        pred_cur = policy_net.mapdecode(h_i_opt.unsqueeze(0))      # [1, (N+1)*4]
        pred_cur_slot = pred_cur.view(1, nagents + 1, 4)

        prey_cur = pred_cur_slot[:, prey_idx, :2]                  # [1,2]
        prey_tgt = target_slot[:, prey_idx, :2]                    # [1,2]

        # 其他实体的 belief
        others_cur = pred_cur_slot[:, :prey_idx, :]                # [1,N,4]
        others_tgt = pred_orig_slot[:, :prey_idx, :]               # 希望维持原值

        loss_target = F.mse_loss(prey_cur, prey_tgt)
        loss_stable = F.mse_loss(others_cur, others_tgt)
        loss_reg = F.mse_loss(h_i_opt, h_i_orig)

        loss = loss_target + 0.1 * loss_stable + 0.01 * loss_reg

        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"[latent opt] step {step+1}/{args.latent_steps}, "
                  f"loss={loss.item():.4f}, "
                  f"prey_cur=({prey_cur[0,0].item():.3f},{prey_cur[0,1].item():.3f})")

    # 返回优化后的 hidden_state_all
    h_all_new = h_all.clone()
    h_all_new[agent_idx] = h_i_opt.detach()
    return h_all, h_all_new, pred_orig_slot, target_slot


def compare_actions(args, policy_net, h_all_orig, h_all_new, agent_idx):
    """
    用原始 hidden 和 幻觉 hidden 分别过 action head，比较该 agent 的动作分布变化。
    """
    nagents = args.nagents
    hid_size = args.hid_size

    h_orig = h_all_orig.view(1, nagents, hid_size)  # [1,N,hid]
    h_new = h_all_new.view(1, nagents, hid_size)

    with torch.no_grad():
        logits_orig = [head(h_orig) for head in policy_net.heads]
        logits_new = [head(h_new) for head in policy_net.heads]

        # 这里假设只有一个离散动作 head，对应 PP 的 move 动作
        logit_o = logits_orig[0][0, agent_idx, :]    # [nactions]
        logit_n = logits_new[0][0, agent_idx, :]

        prob_o = F.softmax(logit_o, dim=-1)
        prob_n = F.softmax(logit_n, dim=-1)

    print("\n=== Action distribution comparison (agent {}) ===".format(agent_idx))
    print("orig logits:", logit_o.detach().cpu().numpy())
    print("new  logits:", logit_n.detach().cpu().numpy())
    print("orig probs :", prob_o.detach().cpu().numpy())
    print("new  probs :", prob_n.detach().cpu().numpy())
    print("argmax orig =", torch.argmax(prob_o).item(),
          ", argmax new =", torch.argmax(prob_n).item())


def main():
    args, _ = parser.parse_known_args()

    args, env, policy_net = load_model_and_env(args)

    # 1. 选一个时间步，拿到 hidden_state / node_decoded / action_out
    chosen = pick_step_and_collect_hidden(args, env, policy_net)
    hidden_state_full = chosen['hidden_state_full']
    node_decoded = chosen['node_decoded']
    agent_idx = args.hallucinate_agent

    seen_ever_prev = chosen['seen_ever_prev']
    seen_ever_now = chosen['seen_ever_now']

    print("\n=== Prey visibility (cumulative) ===")
    print("0..t-1: agents that have ever seen prey:", np.where(seen_ever_prev)[0])
    print("0..t-1: any agent ever saw prey        :", bool(seen_ever_prev.any()))
    print("0..t  : agents that have ever seen prey:", np.where(seen_ever_now)[0])
    print("0..t  : any agent ever saw prey        :", bool(seen_ever_now.any()))

    hidden_state_full = chosen['hidden_state_full']   # [N*1,hid]
    node_decoded = chosen['node_decoded']             # [N,N+1,4]
    agent_idx = args.hallucinate_agent

    print(f"[Hallucination] using agent index = {agent_idx}")

    # 2. 在 latent 空间中对该 agent 的 hidden 做 prey 幻觉注入
    h_all_orig, h_all_new, pred_orig_slot, target_slot = \
        optimize_hidden_for_prey_hallucination(
            args, policy_net, hidden_state_full, node_decoded, agent_idx
        )

    prey_idx = args.nagents
    print("\n=== Belief (prey slot) before & target ===")
    print("orig prey (x,y):",
          pred_orig_slot[0, prey_idx, 0].item(),
          pred_orig_slot[0, prey_idx, 1].item())
    print("target prey (x,y):",
          target_slot[0, prey_idx, 0].item(),
          target_slot[0, prey_idx, 1].item())
    # ============ 打印真实 prey 坐标 和 agent 坐标 ============
    # env.env.prey_loc 是真实 prey 坐标，注意是整数格
    real_prey = env.env.prey_loc[0]  # [x, y]
    agent_pos = env.env.predator_loc[agent_idx]  # [x, y]

    print("\n=== Environment Real Coordinates ===")
    print(f"Agent {agent_idx} real position  :", agent_pos)
    print(f"Real prey position                :", real_prey)

    # ============ 打印注入幻觉后的 predicted prey belief ============
    with torch.no_grad():
        pred_new = policy_net.mapdecode(
            h_all_new[agent_idx].unsqueeze(0)
        ).view(1, args.nagents + 1, 4)

    print("\n=== Belief after hallucination ===")
    print("hallucinated prey (x,y):",
          pred_new[0, prey_idx, 0].item(),
          pred_new[0, prey_idx, 1].item())

    # ============ 打印 hidden state 改变的幅度 ============
    delta_norm = torch.norm(h_all_new[agent_idx] - h_all_orig[agent_idx]).item()
    print("\n=== Hidden State Change ===")
    print("||Δh|| =", delta_norm)

    # 3. 比较动作分布是否发生了显著变化
    compare_actions(args, policy_net, h_all_orig, h_all_new, agent_idx)


if __name__ == "__main__":
    main()