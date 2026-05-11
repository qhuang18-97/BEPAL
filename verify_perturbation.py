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

def load_model_and_data(args):
    # --- 1. 完全复刻 eval_generator_c.py 的初始化逻辑 ---
    # 初始化环境
    env = data.init(args.env_name, args, False)

    # IC3Net 特殊逻辑
    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0

    # 设置输入输出维度
    args.num_inputs = env.observation_dim
    args.num_actions = env.num_actions

    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions

    # 处理 Hard Attention + CommNet 的额外动作维度
    if args.hard_attn and args.commnet:
        if len(args.num_actions) == 1:
            args.num_actions = [*args.num_actions, 2]
            args.dim_actions = env.dim_actions + 1

    # 处理 RNN 设置
    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'

    # 再次调用 parse_action_args 确保万无一失
    parse_action_args(args)

    # --- 2. 初始化网络 ---
    model = CommNetMLP(args, args.num_inputs)

    # --- 3. 加载权重 ---
    if args.load != '':
        if os.path.exists(args.load):
            checkpoint = torch.load(args.load, map_location='cpu')
            if 'policy_net' in checkpoint:
                model.load_state_dict(checkpoint['policy_net'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {args.load}")
        else:
            raise FileNotFoundError(f"Model path not found: {args.load}")

    model.eval()

    # --- 4. 加载数据 ---
    print(f"Loading data from {args.data_path}...")
    dataset = torch.load(args.data_path)

    return model, dataset


def verify_macro(args):
    model, dataset = load_model_and_data(args)

    results_path = 'Probe_case/route_c_macro_results.pt'
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}")
        return
    results = torch.load(results_path)
    print(f"Loaded {len(results)} micro-attack samples.")

    MACRO_MAGNITUDE = 0.15

    verified_count = 0
    total_checked = 0

    print(f"\n--- Extending Micro-Deltas to Macro-Scale ({MACRO_MAGNITUDE}) ---")

    for res in results:
        idx = res['idx']
        micro_delta = torch.tensor(res['delta'])
        target_ent = res['target_entity']
        orig_act = res['orig_act']

        # 1. 拿到原始 h
        h = dataset['h'][idx].unsqueeze(0)

        # --- 准备工作 (无梯度) ---
        target_belief = None
        with torch.no_grad():
            # 2. 计算 Base Belief Map
            base_belief = model.mapdecode(h).view(-1, 4)

            # 3. 构造宏观扰动
            xy_vec = micro_delta[:2]
            current_mag = torch.norm(xy_vec)

            if current_mag < 1e-6:
                continue

            macro_xy = (xy_vec / current_mag) * MACRO_MAGNITUDE
            macro_delta = torch.zeros_like(micro_delta)
            macro_delta[:2] = macro_xy
            macro_delta[2:] = micro_delta[2:]

            # 4. 施加到 Belief 上
            full_delta = torch.zeros_like(base_belief)
            full_delta[target_ent] = macro_delta

            target_belief = base_belief + full_delta
            target_belief[:, :2] = torch.clamp(target_belief[:, :2], 0.0, 1.0)

        # ==========================================
        # 【关键修复】 Step 5 & 6 必须在 no_grad 之外！
        # ==========================================

        # 5. 快速逆向寻找对应的 h_new
        # 我们使用 target_belief (它是 detached 的常量) 作为目标
        h_new = h.clone().detach()
        h_new.requires_grad = True
        opt = optim.Adam([h_new], lr=0.05)

        for _ in range(30):
            opt.zero_grad()
            # 这里 model.mapdecode 需要计算梯度，所以不能在 no_grad 里
            decoded = model.mapdecode(h_new).view(-1, 4)
            # 只计算前 n_ent 个实体的 Loss，避免 padding 干扰
            n_ent = decoded.shape[0]
            loss = F.mse_loss(decoded, target_belief[:n_ent])
            loss.backward()
            opt.step()

        # 6. 检查 Policy 反应
        # 同样，这里也需要梯度或者是正常的 forward
        with torch.no_grad():  # 这里的 forward 不需要梯度，但 h_new 已经是优化好的值了
            logits = model.heads[0](h_new)
            new_act = torch.argmax(logits, dim=1).item()

        total_checked += 1
        if new_act != orig_act:
            verified_count += 1

    print(f"\nVerification Result:")
    print(f"Total Micro-Samples Checked: {total_checked}")
    print(f"Macro-Validated Samples:     {verified_count}")
    if total_checked > 0:
        print(f"Validation Rate:             {verified_count / total_checked:.1%}")

    print("\n[Analysis]")
    if total_checked > 0 and verified_count / total_checked > 0.5:
        print("✅ The micro-deltas represent REAL causal directions.")
        print("   The optimizer was just lazy. Scaling them up works!")
    else:
        print("❌ The micro-deltas were largely ADVERSARIAL NOISE.")
        print("   Scaling them up broke the attack. The policy is fragile/jittery.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify Macro Causality')
    # 基础参数
    parser.add_argument('--env_name', default="predator_prey")
    parser.add_argument('--nagents', type=int, default=5)
    parser.add_argument('--load', default='./saved_model/lec_no_cnn_dl5_d12_r2', type=str)
    parser.add_argument('--data_path', default='./Probe_case/data_key_h_with_gt.pt', type=str)

    # 兼容性参数 (必须要有)
    parser.add_argument('--hid_size', default=64, type=int)
    parser.add_argument('--recurrent', action='store_true', default=True)
    parser.add_argument('--ic3net', action='store_true', default=False)
    parser.add_argument('--commnet', action='store_true', default=False)
    parser.add_argument('--display', action="store_true", default=False)
    parser.add_argument('--obstacles', default=10, type=int)
    parser.add_argument('--comm_passes', type=int, default=1)
    parser.add_argument('--hard_attn', action='store_true', default=False)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--share_weights', default=False, action='store_true')
    parser.add_argument('--comm_init', default='uniform', type=str)

    init_args_for_env(parser)
    args, _ = parser.parse_known_args()

    # 补全
    args.dim = 12
    args.nfriendly = args.nagents

    verify_macro(args)