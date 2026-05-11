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

parser = argparse.ArgumentParser(description='BEPAL PP hallucination experiment')

# ===== 基本训练参数（保持和 main.py 兼容） =====
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--epoch_size', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--nprocesses', type=int, default=1)

# model
parser.add_argument('--hid_size', default=64, type=int)
parser.add_argument('--recurrent', action='store_true', default=False)

# optimization
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

# CommNet specific
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

# ===== 幻觉实验参数 =====
parser.add_argument('--hallucinate_agent', type=int, default=0,
                    help='default agent index; when hallu_mode>0 we会自动覆盖为触发条件的agent')
parser.add_argument('--hallucinate_step', type=int, default=5,
                    help='fallback fixed step when hallu_mode=0')
parser.add_argument('--latent_lr', type=float, default=0.05)
parser.add_argument('--latent_steps', type=int, default=100)

# 条件控制
parser.add_argument('--hallu_mode', type=int, default=0,
                    help='0: fixed step; 1: self sees prey; '
                         '2: teammate saw prey at t-1; 3: inertia after K steps')
parser.add_argument('--inertia_K', type=int, default=3,
                    help='for hallu_mode=3, consecutive steps with prey info')
parser.add_argument('--max_episodes', type=int, default=10,
                    help='max episodes to search for a trigger condition')


def compute_prey_seen(env, args):
    """
    返回每个agent此刻是否能看到prey（基于GT与vision）。
    规则：|dx| <= vision 且 |dy| <= vision → 视为“看到猎物”。
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
    # 初始化 env 相关参数（和 main.py 一样）
    init_args_for_env(parser)
    parsed_args, _ = parser.parse_known_args(sys.argv[1:])
    args = parsed_args

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

    if args.load != '':
        current_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        d = torch.load(current_path + args.load, map_location='cpu')
        policy_net.load_state_dict(d['policy_net'])
        print(f"[Hallucination] loaded model from {current_path + args.load}")

    return args, env, policy_net


def pick_step_and_collect_hidden(args, env, policy_net):
    """
    关键改变：不止一条episode，而是最多跑 max_episodes 条。
    在每条episode内部，根据 hallu_mode & 条件，在任意agent上触发：
      mode 0：固定 step, 使用 args.hallucinate_agent
      mode 1：某个agent自己当前看到prey
      mode 2：某个agent当前没看到，但上一帧有队友看到
      mode 3：某个agent连续K步有prey信息（惯性）

    一旦命中条件，就返回那个 episode 中触发时刻 t 的：
      - state_t
      - hidden_state_t
      - action_out_t
      - node_decoded_t
      - agent_idx（触发的那个agent）
    """
    nagents = args.nagents
    prey_idx = nagents

    last_snapshot = None  # 如果所有episode都没触发，用它做fallback

    for ep in range(args.max_episodes):
        # ===== reset 一条新的 episode =====
        reset_args = getargspec(env.reset).args
        if 'epoch' in reset_args:
            state, action_mask = env.reset(0)
        else:
            state, action_mask = env.reset()

        info = {}
        prev_hid = torch.zeros(1, nagents, args.hid_size)

        # 累计看到情况（在当前episode内部）
        seen_ever = np.zeros(nagents, dtype=bool)
        seen_prev = np.zeros(nagents, dtype=bool)

        # mode=3：每个agent自己的惯性计数
        consec_seen = np.zeros(nagents, dtype=int)

        for t in range(args.max_steps):
            # 1. 计算当前瞬时谁看到prey
            seen_now = compute_prey_seen(env, args)
            seen_ever_prev = seen_ever.copy()
            seen_ever_now = seen_ever | seen_now

            if t == 0 and args.hard_attn and args.commnet:
                info['comm_action'] = np.zeros(nagents, dtype=int)

            if args.recurrent:
                if args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = policy_net.init_hidden(batch_size=state.shape[0])
                x = [state, prev_hid]
                with torch.no_grad():
                    action_out, value, value_global, prev_hid, node_decoded = policy_net(x, info)
            else:
                raise RuntimeError("当前实验脚本假设使用 recurrent=LSTM")

            hidden_state_full = prev_hid[0].detach().clone()
            node_decoded = node_decoded.view(nagents,
                                             nagents + 1,
                                             -1).detach().clone()

            # 记录一份 snapshot，万一所有episode都没触发用作fallback
            last_snapshot = {
                't': t,
                'state': state.clone(),
                'info': dict(info),
                'hidden_state_full': hidden_state_full,
                'action_out': [a.clone() for a in action_out],
                'node_decoded': node_decoded.clone(),
                'seen_ever_prev': seen_ever_prev.copy(),
                'seen_ever_now': seen_ever_now.copy(),
                'seen_now': seen_now.copy(),
                'seen_prev': seen_prev.copy(),
                'episode_idx': ep,
                'agent_idx': args.hallucinate_agent,  # fallback使用
                'trigger_reason': f"fallback_ep{ep}_t{t}"
            }

            trigger = False
            trigger_reason = ""
            chosen_agent = None

            # 2. 根据 hallu_mode 在所有agents上扫描触发
            if args.hallu_mode == 0:
                # 固定 step + 固定 agent
                if t == args.hallucinate_step:
                    trigger = True
                    trigger_reason = f"fixed_step_{t}"
                    chosen_agent = args.hallucinate_agent

            elif args.hallu_mode == 1:
                # 自己当前看到prey：找任意 seen_now[i] == True 的agent
                # if t >= 10:
                #     # “第一次看到”的定义：之前从未seen_ever_prev[i]，现在 seen_now[i] == True
                #     first_time_indices = np.where((seen_now == True) & (seen_ever_prev == False))[0]
                #     if len(first_time_indices) > 0:
                #         chosen_agent = int(first_time_indices[0])
                if True:
                    candidates = np.where(seen_now)[0]
                    if len(candidates) > 0:
                        chosen_agent = int(candidates[0])
                        trigger = True
                        trigger_reason = f"self_seen_agent{chosen_agent}_t{t}"

            elif args.hallu_mode == 2:
                # 队友上一步看到prey，这一步自己没看到
                if True:#t>10:
                    for i in range(nagents):
                        teammate_saw_prev = np.any(seen_prev[np.arange(nagents) != i])
                        if teammate_saw_prev and (not seen_now[i]):
                            chosen_agent = i
                            trigger = True
                            trigger_reason = f"teammate_prev_seen_agent{i}_t{t}"
                            break

            elif args.hallu_mode == 3:
                # 惯性：每个agent维护自己的连续看到计数
                if t>10:
                    for i in range(nagents):
                        if seen_now[i]:
                            consec_seen[i] += 1
                        else:
                            consec_seen[i] = 0

                    # 找任何一个 consec_seen[i] >= K 的agent
                    candidates = np.where(consec_seen >= args.inertia_K)[0]
                    if len(candidates) > 0:
                        chosen_agent = int(candidates[0])
                        trigger = True
                        trigger_reason = f"inertia_agent{chosen_agent}_consec{consec_seen[chosen_agent]}_t{t}"

            if trigger:
                chosen = {
                    't': t,
                    'state': state.clone(),
                    'info': dict(info),
                    'hidden_state_full': hidden_state_full,
                    'action_out': [a.clone() for a in action_out],
                    'node_decoded': node_decoded.clone(),
                    'seen_ever_prev': seen_ever_prev.copy(),
                    'seen_ever_now': seen_ever_now.copy(),
                    'seen_now': seen_now.copy(),
                    'seen_prev': seen_prev.copy(),
                    'episode_idx': ep,
                    'agent_idx': chosen_agent,
                    'trigger_reason': trigger_reason
                }
                print(f"[Hallucination] TRIGGER at episode {ep}, t = {t}, "
                      f"agent = {chosen_agent}, reason = {trigger_reason}")
                return chosen

            # 3. rollout 前进一步
            action = select_action(args, action_out)
            action, actual = translate_action(args, env, action)
            next_state, action_mask, reward, done, info = env.step(actual)

            if args.hard_attn and args.commnet:
                if not args.comm_action_one:
                    info['comm_action'] = action[-1]
                else:
                    info['comm_action'] = np.ones(nagents, dtype=int)

            state = next_state
            seen_ever = seen_ever_now
            seen_prev = seen_now

            if done:
                break

        # 这一条 episode 没触发，进入下一条 episode
        print(f"[Hallucination] No trigger in episode {ep}, resetting env.")

    # 如果所有 episodes 都没触发，fallback
    print("[Hallucination] WARNING: no trigger in all episodes, using fallback snapshot.")
    return last_snapshot


def optimize_hidden_for_prey_hallucination(args, policy_net, hidden_state_full, node_decoded, agent_idx):
    """
    对 chosen agent 的 hidden 做latent优化：
      目标：把mapdecode输出的prey (x,y)镜像到对角线另一侧：
        target_x = 1 - x_orig
        target_y = 1 - y_orig
    同时尽量保持其它实体belief不变 + h不要偏离太多。
    """
    nagents = args.nagents
    hid_size = args.hid_size
    prey_idx = nagents

    h_all = hidden_state_full.view(nagents, hid_size)
    h_i_orig = h_all[agent_idx].detach()
    h_i_opt = h_i_orig.clone().detach().requires_grad_(True)

    with torch.no_grad():
        pred_orig = policy_net.mapdecode(h_i_orig.unsqueeze(0))  # [1,(N+1)*4]
    pred_orig_slot = pred_orig.view(1, nagents + 1, 4)

    x_orig = pred_orig_slot[0, prey_idx, 0].item()
    y_orig = pred_orig_slot[0, prey_idx, 1].item()

    target_x = 1.0 - x_orig
    target_y =  1.0 - y_orig

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

        if (step + 1) % 20 == 0:
            print(f"[latent opt] step {step+1}/{args.latent_steps}, "
                  f"loss={loss.item():.4f}, "
                  f"prey_cur=({prey_cur[0,0].item():.3f},{prey_cur[0,1].item():.3f})")

    h_all_new = h_all.clone()
    h_all_new[agent_idx] = h_i_opt.detach()
    return h_all, h_all_new, pred_orig_slot, target_slot


def compare_actions(args, policy_net, h_all_orig, h_all_new, agent_idx):
    nagents = args.nagents
    hid_size = args.hid_size

    h_orig = h_all_orig.view(1, nagents, hid_size)
    h_new = h_all_new.view(1, nagents, hid_size)

    with torch.no_grad():
        logits_orig = [head(h_orig) for head in policy_net.heads]
        logits_new = [head(h_new) for head in policy_net.heads]

        logit_o = logits_orig[0][0, agent_idx, :]
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

    # 1. 多episode搜索触发时刻 + agent
    chosen = pick_step_and_collect_hidden(args, env, policy_net)
    hidden_state_full = chosen['hidden_state_full']
    node_decoded = chosen['node_decoded']
    agent_idx = chosen['agent_idx']

    seen_ever_prev = chosen['seen_ever_prev']
    seen_ever_now = chosen['seen_ever_now']

    print(f"\n[Hallucination] from episode {chosen['episode_idx']}, t = {chosen['t']}")
    print("\n=== Prey visibility (cumulative) ===")
    print("0..t-1: agents that have ever seen prey:", np.where(seen_ever_prev)[0])
    print("0..t-1: any agent ever saw prey        :", bool(seen_ever_prev.any()))
    print("0..t  : agents that have ever seen prey:", np.where(seen_ever_now)[0])
    print("0..t  : any agent ever saw prey        :", bool(seen_ever_now.any()))

    print("trigger_reason:", chosen['trigger_reason'])
    print("seen_prev (instant):", np.where(chosen['seen_prev'])[0])
    print("seen_now  (instant):", np.where(chosen['seen_now'])[0])
    print(f"[Hallucination] using agent index = {agent_idx}")

    # 2. latent 幻觉注入
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

    real_prey = env.env.prey_loc[0]
    agent_pos = env.env.predator_loc[agent_idx]

    print("\n=== Environment Real Coordinates ===")
    print(f"Agent {agent_idx} real position  :", agent_pos)
    print(f"Real prey position                :", real_prey)
    # === Obstacle info ===
    obs_grid = env.env.obstacle_grid  # shape [M, M], 0/1
    obs_coords = np.argwhere(obs_grid == 1)  # 每个元素是 [row, col]

    print("\n=== Obstacles (row, col) ===")
    print(f"Total obstacles: {len(obs_coords)}")
    print(obs_coords)
    with torch.no_grad():
        pred_new = policy_net.mapdecode(
            h_all_new[agent_idx].unsqueeze(0)
        ).view(1, args.nagents + 1, 4)

    print("\n=== Belief after hallucination ===")
    print("hallucinated prey (x,y):",
          pred_new[0, prey_idx, 0].item(),
          pred_new[0, prey_idx, 1].item())

    delta_norm = torch.norm(h_all_new[agent_idx] - h_all_orig[agent_idx]).item()
    print("\n=== Hidden State Change ===")
    print("||Δh|| =", delta_norm)

    compare_actions(args, policy_net, h_all_orig, h_all_new, agent_idx)


if __name__ == "__main__":
    main()
