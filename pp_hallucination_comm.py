# pp_comm_injection.py
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


parser = argparse.ArgumentParser(description='BEPAL PP communication hallucination (Stage 3)')

# ===== 基本训练 / 环境参数（直接照 main.py / pp_hallucination 那一套）=====
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
parser.add_argument(
    '--min_t_trigger',
    type=int,
    default=8,
    help='minimal timestep t for communication hallucination trigger'
)
parser.add_argument('--max_episodes', type=int, default=50,
                    help='max episodes to search for a valid comm hallucination trigger')

# 观测范围（和 env 里 vision 对齐）
# parser.add_argument('--vision', type=int, default=2)

# 幻觉优化参数
parser.add_argument('--latent_lr', type=float, default=0.05)
parser.add_argument('--latent_steps', type=int, default=100)


def compute_prey_seen(env, args):
    """
    判断在当前 env 状态下，每个 agent 是否能看到 prey。
    这里按照 vision 范围做一个简单判定：
      abs(dx) <= vision 且 abs(dy) <= vision → 认为“看到猎物”
    注意：这里用的是环境里的真实坐标 (row, col)。
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


def load_model_and_env():
    """初始化 env + policy_net，并加载已训练模型。"""
    init_args_for_env(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])

    # IC3Net: 强制打开 commnet + hard_attn
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

    # 加载模型
    if args.load != '':
        current_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
        d = torch.load(current_path + args.load, map_location='cpu')
        policy_net.load_state_dict(d['policy_net'])
        print(f"[CommHallucination] loaded model from {current_path + args.load}")

    return args, env, policy_net


def optimize_sender_hidden_for_fake_prey(args, policy_net, prev_hid, sender_idx,
                                         target_x, target_y):
    """
    给定:
      - prev_hid: (hidden_state, cell_state)，形状 [N, hid_size]
      - sender_idx: 发消息的 agent index
      - target_x, target_y: 目标 prey 归一化坐标（0~1），这里来自 GT 反向坐标

    做的事:
      - 只优化 sender 的 hidden h_sender
      - 让 mapdecode(h_sender) 的 prey-slot 的 (x,y) 靠近 (target_x, target_y)
      - 保持其他 slot 尽量不动 + 正则限制 h_sender 不要离原始太远
    """
    nagents = args.nagents
    hid_size = args.hid_size
    prey_idx = nagents  # 最后一个 slot 是 prey（和之前脚本保持一致）

    hidden_state, cell_state = prev_hid  # [N*B, hid_size]，这里 B=1
    h_all = hidden_state.view(nagents, hid_size)  # [N, hid_size]

    h_i_orig = h_all[sender_idx].detach()
    h_i_opt = h_i_orig.clone().detach().requires_grad_(True)

    # 原始 decoder 输出（只用于 “other slots 保持稳定”）
    with torch.no_grad():
        pred_orig = policy_net.mapdecode(h_i_orig.unsqueeze(0))  # [1, (N+1)*4]
    pred_orig_slot = pred_orig.view(1, nagents + 1, 4)          # [1, N+1, 4]

    # 用 GT 反向坐标作为目标（不再用 orig belief）
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

    # 把优化后的 h_i 塞回整条 hidden
    h_all_new = h_all.clone()
    h_all_new[sender_idx] = h_i_opt.detach()

    prev_hid_poison = (
        h_all_new.view_as(hidden_state).detach(),
        cell_state.clone().detach()
    )

    # 一点 debug 信息
    with torch.no_grad():
        pred_new = policy_net.mapdecode(h_all_new[sender_idx].unsqueeze(0)).view(
            1, nagents + 1, 4
        )
    prey_new = (pred_new[0, prey_idx, 0].item(),
                pred_new[0, prey_idx, 1].item())

    delta_h = torch.norm(h_all_new[sender_idx] - h_all[sender_idx]).item()

    debug_info = {
        'delta_h': delta_h,
        'prey_new': prey_new,
    }

    return prev_hid_poison, debug_info

def to_comm_action_1d(comm_raw, nagents):
    """
    把 comm_action 规范成长度为 nagents 的 1D numpy 向量（0/1）。
    支持输入是 torch.Tensor / numpy / list / shape 带 batch 维等情况。
    """
    import numpy as np
    import torch

    if isinstance(comm_raw, torch.Tensor):
        comm = comm_raw.detach().cpu().numpy()
    else:
        comm = np.array(comm_raw)

    # 压扁，只保留一维 [nagents]
    comm = comm.reshape(-1)

    # 有时候可能多出来元素（比如带 action head 维度），只取前 nagents 个
    if comm.shape[0] > nagents:
        comm = comm[:nagents]

    return comm


def run_comm_injection(args, env, policy_net):
    """
    多 episode 搜索一个满足条件的 t→t+1 通信注入案例：

      条件：
        - t >= min_t_trigger
        - 当前时刻 t 恰好只有 1 个 agent 能看到 prey
          （这个 agent 作为 sender，其余看不到的作为 receivers）

      流程：
        对 ep in [0, max_episodes):
          1) reset 环境和 prev_hid
          2) 在该 episode 内，对 t 从 0..max_steps-2：
               a) 计算 seen_now
               b) 若 num_seen == 1 且 t >= min_t_trigger：
                    i.   用干净 prev_hid_t 正常走一步 → 得到 prev_hid_{t+1}, next_state
                    ii.  用 GT 构造 fake prey norm 坐标
                    iii. 在 prev_hid_{t+1} 上只篡改 sender 的 hidden
                    iv.  在 t+1 比较干净 vs 下毒的 action（sender 和 receivers）
                    v.   打印完就 return
          3) 如果这一局没触发，就继续下一局

        如果所有 episode 都没触发，打印总的 warning。
    """
    triggered_global = False

    for ep in range(args.max_episodes):
        print(f"\n[CommHallucination] === Episode {ep} ===")

        # ---- reset env + hidden 每一局都要重置 ----
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
            raise RuntimeError("当前脚本假设使用 recurrent (LSTM/IC3Net) 模式。")

        triggered_this_ep = False

        # 留出 t+1，所以到 max_steps-1
        for t in range(args.max_steps - 1):
            seen_now = compute_prey_seen(env, args)
            num_seen = np.sum(seen_now)

            # 第一步需要初始化 comm_action
            if args.hard_attn and args.commnet and t == 0:
                info['comm_action'] = np.zeros(nagents, dtype=int)

            # ====== 触发条件：只有一个 agent 看到 prey，且 t >= min_t_trigger ======
            if (not triggered_this_ep) and (num_seen == 1) and (t >= args.min_t_trigger):
                sender_idx = int(np.where(seen_now)[0][0])
                receivers = np.where(~seen_now)[0].tolist()

                prey_loc = np.array(env.env.prey_loc[0])  # [row, col]
                M = env.env.dims[0]

                # 真实绝对坐标标准化: x=col/(M-1), y=row/(M-1)
                x_real = prey_loc[0] / (M - 1)
                y_real = prey_loc[1] / (M - 1)

                # 反向假坐标
                target_x = 1.0 - x_real
                target_y = 1.0 - y_real

                print(f"[CommHallucination] Trigger at episode {ep}, t = {t}")
                print(f"  sender    = {sender_idx}")
                print(f"  receivers = {receivers}")
                print(f"  real prey (row,col) = {prey_loc}")
                print(f"  real prey norm (x,y)= ({x_real:.3f}, {y_real:.3f})")
                print(f"  fake prey norm (x,y)= ({target_x:.3f}, {target_y:.3f})")
                # === Obstacle info ===
                obs_grid = env.env.obstacle_grid  # shape [M, M], 0/1
                obs_coords = np.argwhere(obs_grid == 1)  # 每个元素是 [row, col]

                print("\n=== Obstacles (row, col) ===")
                print(f"Total obstacles: {len(obs_coords)}")
                print(obs_coords)

                # 打印 t 时刻所有 agent 位置
                agent_pos_all_t = np.array(env.env.predator_loc)
                print("  agent positions at t (row,col):")
                for i in range(nagents):
                    print(f"    agent {i}: {agent_pos_all_t[i]}")

                # ===== 1) t 时刻：用干净 prev_hid_t 正常走一步，得到 t+1 的 state & hidden =====
                x_t = [state, prev_hid]
                with torch.no_grad():
                    action_out_t, value_t, valueg_t, prev_hid_next, node_dec_t = \
                        policy_net(x_t, info)

                # 选动作 + step 环境
                action_t = select_action(args, action_out_t)
                action_exec_t, actual_t = translate_action(args, env, action_t)
                next_state, action_mask_next, reward, done, info_next = env.step(actual_t)

                # t+1 的 comm_action
                if args.hard_attn and args.commnet:
                    if not args.comm_action_one:
                        comm_raw = action_t[-1]
                        info_next['comm_action'] = to_comm_action_1d(comm_raw, nagents)
                    else:
                        info_next['comm_action'] = np.ones(nagents, dtype=int)

                # 打印 t+1 时刻真实位置
                prey_loc_t1 = np.array(env.env.prey_loc[0])
                agent_pos_all_t1 = np.array(env.env.predator_loc)
                print(f"[CommHallucination] State at t+1 before injection:")
                print(f"  prey (row,col) = {prey_loc_t1}")
                print("  agent positions at t+1 (row,col):")
                for i in range(nagents):
                    print(f"    agent {i}: {agent_pos_all_t1[i]}")

                print("[CommHallucination] Clean step t done, now injecting into t+1 hidden.")

                # ===== 2) 在 prev_hid_{t+1} 上篡改 sender 的 hidden → prev_hid_poison_{t+1} =====
                prev_hid_poison, debug = optimize_sender_hidden_for_fake_prey(
                    args, policy_net, prev_hid_next, sender_idx, target_x, target_y
                )
                print(f"[CommHallucination] ||Δh_sender|| = {debug['delta_h']:.4f}")
                print(f"[CommHallucination] prey slot after hallucination (x,y) = "
                      f"({debug['prey_new'][0]:.3f}, {debug['prey_new'][1]:.3f})")

                # ===== 3) t+1：同一个 next_state 下，对比干净 vs 下毒 的 action =====
                x_t1_clean = [next_state, prev_hid_next]
                x_t1_poison = [next_state, prev_hid_poison]

                with torch.no_grad():
                    action_clean, value_c, valueg_c, _, node_c = \
                        policy_net(x_t1_clean, info_next)
                    action_poison, value_p, valueg_p, _, node_p = \
                        policy_net(x_t1_poison, info_next)

                logp_clean = action_clean[0][0]   # [N, A]
                logp_poison = action_poison[0][0]
                prob_clean = torch.exp(logp_clean)
                prob_poison = torch.exp(logp_poison)

                # sender
                print("\n=== Action distribution at t+1 (sender) ===")
                print(f"sender idx = {sender_idx}")
                print("orig logits:", logp_clean[sender_idx].detach().cpu().numpy())
                print("new  logits:", logp_poison[sender_idx].detach().cpu().numpy())
                print("orig probs :", prob_clean[sender_idx].detach().cpu().numpy())
                print("new  probs :", prob_poison[sender_idx].detach().cpu().numpy())
                print("argmax orig =", int(torch.argmax(prob_clean[sender_idx]).item()),
                      ", argmax new =", int(torch.argmax(prob_poison[sender_idx]).item()))

                # receivers
                print("\n=== Action distribution at t+1 (receivers) ===")
                for ridx in receivers:
                    print(f"\n[Receiver {ridx}]")
                    print("orig logits:", logp_clean[ridx].detach().cpu().numpy())
                    print("new  logits:", logp_poison[ridx].detach().cpu().numpy())
                    print("orig probs :", prob_clean[ridx].detach().cpu().numpy())
                    print("new  probs :", prob_poison[ridx].detach().cpu().numpy())
                    print("argmax orig =", int(torch.argmax(prob_clean[ridx]).item()),
                          ", argmax new =", int(torch.argmax(prob_poison[ridx]).item()))


                # 假设 dim=4 对应 [Row, Col, vRow, vCol] (或者 [x, y, vx, vy])
                # 具体取决于 trainer.py 的定义，这里我们打印前两位
                beliefs_poison = node_p.view(nagents, nagents + 1, 4)
                beliefs_clean = node_c.view(nagents, nagents + 1, 4)

                # Sender 的 Belief 也可以顺便打印一下，确认它自己是不是也被“视觉”修正回去了
                print("\n=== Belief (Prey Slot) at t+1 (sender) ===")
                sender_belief = beliefs_poison[sender_idx, nagents, :2].detach().cpu().numpy()
                print(f"sender idx = {sender_idx}")
                print(f"prey belief (val0, val1): ({sender_belief[0]:.3f}, {sender_belief[1]:.3f})")

                # Receivers 的 Action 和 Belief
                print("\n=== Action & Belief at t+1 (receivers) ===")
                for ridx in receivers:
                    print(f"\n[Receiver {ridx}]")

                    # 1. Action 分布对比 (原有的)
                    print("Action Probs:")
                    print("  orig :", prob_clean[ridx].detach().cpu().numpy())
                    print("  new  :", prob_poison[ridx].detach().cpu().numpy())
                    print("  argmax: {} -> {}".format(
                        int(torch.argmax(prob_clean[ridx]).item()),
                        int(torch.argmax(prob_poison[ridx]).item())
                    ))

                    # 2. 【新增】Belief Decoder 输出对比 (验证通信语义)
                    # nagents 是 prey 的 slot index
                    b_clean = beliefs_clean[ridx, nagents, :2].detach().cpu().numpy()
                    b_poison = beliefs_poison[ridx, nagents, :2].detach().cpu().numpy()

                    # 计算 Belief 的偏移量 (Shift)
                    belief_shift = np.linalg.norm(b_poison - b_clean)

                    print("Belief Decoder (Prey Slot):")
                    print(f"  orig belief : ({b_clean[0]:.3f}, {b_clean[1]:.3f})")
                    print(f"  new  belief : ({b_poison[0]:.3f}, {b_poison[1]:.3f})")
                    print(f"  belief shift: {belief_shift:.4f}")

                    # 简单判断：Receiver 是否“看到”了幻觉？
                    # 比较 new belief 和 target (fake prey)
                    # 注意：这里需要你传入 target_x, target_y 或者在外面定义过
                    # 假设 target_x (Col), target_y (Row) 是你攻击的目标
                    # 如果 Decoder 是 (Row, Col)，则比较 (b_poison[0], b_poison[1]) 和 (target_y, target_x)
                    dist_to_fake = np.sqrt((b_poison[0] - target_y) ** 2 + (b_poison[1] - target_x) ** 2)
                    print(f"  dist to fake target: {dist_to_fake:.4f}")


                print("\n[CommHallucination] Finished one t→t+1 injection experiment, stop here.")
                triggered_this_ep = True
                triggered_global = True
                return  # 找到一个案例就退出整个函数

            # ===== 如果这一 timestep 还没触发，就正常滚动一步，继续找 =====
            x = [state, prev_hid]
            with torch.no_grad():
                action_out, value, value_global, prev_hid, node_decoded = \
                    policy_net(x, info)

            action = select_action(args, action_out)
            action_exec, actual = translate_action(args, env, action)
            next_state, action_mask, reward,done, info = env.step(actual)

            if args.hard_attn and args.commnet:
                if not args.comm_action_one:
                    comm_raw = action[-1]
                    info['comm_action'] = to_comm_action_1d(comm_raw, nagents)
                else:
                    info['comm_action'] = np.ones(nagents, dtype=int)

            state = next_state

        if not triggered_this_ep:
            print(f"[CommHallucination] No valid trigger in episode {ep}, resetting env...")

    if not triggered_global:
        print("[CommHallucination] WARNING: no episode produced a valid trigger "
              f"within max_episodes = {args.max_episodes}.")




def main():
    args, env, policy_net = load_model_and_env()
    run_comm_injection(args, env, policy_net)


if __name__ == "__main__":
    main()
