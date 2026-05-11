import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import data
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args

# 确保双精度
torch.set_default_tensor_type('torch.DoubleTensor')


def load_model(args):
    # --- 标准的模型加载逻辑 (与之前一致) ---
    defaults = {
        'nfriendly': getattr(args, 'nagents', 5),
        'continuous': False, 'hard_attn': False, 'commnet': False, 'ic3net': False,
        'comm_mode': 'avg', 'comm_passes': 1, 'comm_mask_zero': False, 'mean_ratio': 1.0,
        'rnn_type': 'MLP', 'detach_gap': 10000, 'comm_init': 'uniform',
        'comm_action_one': False, 'advantages_per_action': False, 'share_weights': False
    }
    for key, val in defaults.items():
        if not hasattr(args, key): setattr(args, key, val)

    if args.ic3net:
        args.commnet = True;
        args.hard_attn = True;
        args.mean_ratio = 0

    env = data.init(args.env_name, args, False)
    args.num_inputs = env.observation_dim
    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)): args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    if args.hard_attn and args.commnet:
        if len(args.num_actions) == 1: args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1
    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True;
        args.rnn_type = 'LSTM'

    parse_action_args(args)
    model = CommNetMLP(args, args.num_inputs)
    if args.load != '':
        checkpoint = torch.load(args.load, map_location='cpu')
        model.load_state_dict(checkpoint['policy_net'] if 'policy_net' in checkpoint else checkpoint)
    model.eval()
    return model, env


def run_route_e(args):
    print(f"Loading data from {args.data_path}...")
    dataset = torch.load(args.data_path)
    h_data = dataset['h'];
    gt_data = dataset['gt'];
    action_data = dataset['action']

    model, _ = load_model(args)

    # --- 实验参数设置 ---
    # 并行尝试次数 (相当于原来的外层循环)
    # 32 意味着对于每个样本，我们同时尝试 32 种不同的物理干预
    BATCH_SIZE = 32

    # 内层优化步数 (Pure Inversion)
    # 只要画得像就行，不需要太多步
    INVERSION_STEPS = 50

    # 物理门槛
    GRID_SIZE = 1.0 / 12.0
    MIN_MAGNITUDE = GRID_SIZE * 1.5  # 至少移动 1.5 个格子

    success_log = []

    # 采样
    num_samples = min(args.num_experiments, len(h_data))
    indices = np.random.choice(len(h_data), num_samples, replace=False)

    print(f"\n--- Starting Route E (Bilevel Decoupled Optimization) ---")
    print(f"Strategy: Batch Monte Carlo Search ({BATCH_SIZE} trials/sample)")
    print(f"Constraint: Pure MSE Inversion (No Attack Loss)")
    print(f"Physical: Delta >= {MIN_MAGNITUDE:.3f}, Bounds [0,1]")

    for i, idx in enumerate(indices):
        # 1. 准备数据: [1, Dim] -> [Batch, Dim]
        h_orig = h_data[idx].view(1, -1)
        orig_act = action_data[idx].item()

        # 扩展成 Batch，并行处理 32 个假设
        h_batch = h_orig.repeat(BATCH_SIZE, 1)

        # 2. 批量生成物理扰动 (Vectorized Perturbation)
        with torch.no_grad():
            # 解码原始画面
            base_belief = model.mapdecode(h_orig).view(-1, 4)  # [N_ent, 4]
            n_ent = base_belief.shape[0]

            # 复制成 Batch: [Batch, N_ent, 4]
            target_beliefs = base_belief.unsqueeze(0).repeat(BATCH_SIZE, 1, 1)

            # A. 随机选人: 0-4 是队友, 5 是 Prey (假设最多6个实体有效)
            # 我们随机让每个 Batch 选一个不同的目标
            target_ents = torch.randint(0, min(n_ent, 6), (BATCH_SIZE,))

            # B. 随机生成力向量 (极坐标)
            # 角度: 0 ~ 2pi
            angles = torch.rand(BATCH_SIZE) * 2 * np.pi
            # 力度: MIN_MAG ~ 0.4 (避免移太远变成异常值)
            mags = torch.rand(BATCH_SIZE) * (0.4 - MIN_MAGNITUDE) + MIN_MAGNITUDE

            dx = mags * torch.cos(angles)
            dy = mags * torch.sin(angles)

            # C. 应用扰动 (Advanced Indexing)
            batch_indices = torch.arange(BATCH_SIZE)

            # 记录原始位置用于后续计算实际位移
            orig_positions = target_beliefs[batch_indices, target_ents, :2].clone()

            # 施加 Delta
            target_beliefs[batch_indices, target_ents, 0] += dx
            target_beliefs[batch_indices, target_ents, 1] += dy

            # D. 物理边界截断 (Clamp)
            # 这一步至关重要，防止优化器去拟合界外坐标
            target_beliefs[:, :, :2] = torch.clamp(target_beliefs[:, :, :2], 0.0, 1.0)

            # E. 再次检查实际位移 (因为 Clamp 可能会把位移吃掉)
            # 如果被墙挡住了，实际位移可能很小，这种样本不算数
            actual_new_pos = target_beliefs[batch_indices, target_ents, :2]
            actual_displacement = torch.norm(actual_new_pos - orig_positions, dim=1)

            # 生成一个 mask，标记哪些 trial 是物理上有效的 (移动够大)
            valid_physics_mask = actual_displacement >= MIN_MAGNITUDE

        # 3. 批量纯净反演 (Batch Pure Inversion)
        # 这里的 h_opt 是我们要优化的对象
        h_opt = h_batch.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([h_opt], lr=0.1)  # LR 可以大一点，因为是拟合

        for _ in range(INVERSION_STEPS):
            optimizer.zero_grad()

            # Decoder
            decoded = model.mapdecode(h_opt).view(BATCH_SIZE, -1, 4)

            # 只计算前 n_ent 个实体的 MSE
            # [Batch, N_ent, 4]
            decoded_valid = decoded[:, :n_ent, :]
            target_valid = target_beliefs

            loss = F.mse_loss(decoded_valid, target_valid)

            loss.backward()
            optimizer.step()

        # 4. 批量验证 (Batch Verification)
        # 到了这一步，h_opt 已经变成了"看起来像被移动过的样子"
        # 关键时刻：Policy 会变吗？
        with torch.no_grad():
            logits = model.heads[0](h_opt)
            new_acts = torch.argmax(logits, dim=1)  # [Batch]

            # 检查是否改变
            flipped_mask = (new_acts != orig_act)

            # 最终成功的条件：动作变了 AND 物理移动够大
            success_mask = flipped_mask & valid_physics_mask

            if success_mask.any():
                # 只要这 32 个尝试里有一个成功，就算发现了因果
                # 我们取第一个成功的
                first_succ_idx = torch.where(success_mask)[0][0].item()

                res = {
                    'idx': idx,
                    'orig_act': orig_act,
                    'new_act': new_acts[first_succ_idx].item(),
                    'target_entity': target_ents[first_succ_idx].item(),
                    'delta_mag': actual_displacement[first_succ_idx].item(),
                    # 记录具体的位移向量
                    'delta_vec': (actual_new_pos[first_succ_idx] - orig_positions[first_succ_idx]).cpu().numpy()
                }
                success_log.append(res)
                # print(f"Sample {idx}: Found Causal! Moved Ent {res['target_entity']} -> Act {res['new_act']}")

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} | Robust Discovery Rate: {len(success_log) / (i + 1):.2%}")

    print(f"\n--- Route E Finished ---")
    print(f"Total Samples: {num_samples}")
    print(f"Robust Causal Samples Found: {len(success_log)} ({len(success_log) / num_samples:.2%})")

    # 保存结果供后续分析
    torch.save(success_log, 'Probe_case/route_e_results.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument('--env_name', default="predator_prey")
    parser.add_argument('--nagents', type=int, default=5)
    parser.add_argument('--load', default='./saved_model/lec_no_cnn_dl5_d12_r2', type=str)
    parser.add_argument('--data_path', default='./Probe_case/data_key_h_with_gt.pt', type=str)
    parser.add_argument('--num_experiments', type=int, default=100)

    # 兼容参数
    parser.add_argument('--hid_size', default=64, type=int)
    parser.add_argument('--recurrent', action='store_true', default=True)
    parser.add_argument('--ic3net', action='store_true', default=False)
    parser.add_argument('--commnet', action='store_true', default=False)
    parser.add_argument('--display', action="store_true", default=False)
    parser.add_argument('--obstacles', default=10, type=int)
    parser.add_argument('--comm_passes', type=int, default=1)
    parser.add_argument('--hard_attn', action='store_true', default=False)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)  # 这里的 batch_size 是模型原本的参数，不要混淆
    parser.add_argument('--share_weights', default=False, action='store_true')
    parser.add_argument('--comm_init', default='uniform', type=str)

    init_args_for_env(parser)
    args, _ = parser.parse_known_args()
    args.dim = 12;
    args.nfriendly = args.nagents

    run_route_e(args)