import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import data
from comm import CommNetMLP
from utils import init_args_for_env
from action_utils import parse_action_args

torch.set_default_tensor_type('torch.DoubleTensor')

# ====== 配置区域 ======
OUTPUT_DIR = "eval_results_Prob_Distribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_SIZE = 12
OFFSET_SCALE_FACTOR = 5.0
ACTION_NAMES = ['Up', 'Right', 'Down', 'Left', 'Stay']  # 对应 0,1,2,3,4


# --- 1. 模型定义 (保持不变) ---
class DualStageGenerator(nn.Module):
    def __init__(self, hidden_dim, num_actions):
        super(DualStageGenerator, self).__init__()
        input_dim = hidden_dim + num_actions
        self.selector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )
        self.steerer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, h, target_action_idx, num_actions_total, temperature=1.0, hard=False):
        batch_size = h.size(0)
        action_one_hot = torch.zeros(batch_size, num_actions_total).to(h.device)
        action_one_hot.scatter_(1, target_action_idx.unsqueeze(1), 1)
        gen_input = torch.cat([h, action_one_hot], dim=1)

        mask_logits = self.selector(gen_input)
        if self.training:
            soft_mask = torch.sigmoid(mask_logits)
        else:
            soft_mask = torch.sigmoid(mask_logits)

        m_hard = (soft_mask > 0.5).float()
        mask = m_hard - soft_mask.detach() + soft_mask

        delta_raw = self.steerer(gen_input)
        delta_final = mask * delta_raw

        return delta_final, mask


def plot_comparison_with_probs(belief, gt, obstacles, probs_orig, probs_new, agent_id, step, title, save_path):
    # 转 Numpy
    if isinstance(belief, torch.Tensor): belief = belief.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor): gt = gt.detach().cpu().numpy()
    if isinstance(obstacles, torch.Tensor): obstacles = obstacles.detach().cpu().numpy()

    n_entities = belief.shape[0]
    n_agents = n_entities - 1

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    ax_map = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ==================== 1. 画地图 (修正版) ====================
    # 还原数值
    b_raw = belief[:, :2] * (GRID_SIZE - 1)
    g_raw = gt[:, :2] * (GRID_SIZE - 1)

    # 【Step 1: 交换 XY】 (Row -> Y, Col -> X)
    # 数据格式是 [Row, Col]
    b_x, b_y = b_raw[:, 1], b_raw[:, 0]
    g_x, g_y = g_raw[:, 1], g_raw[:, 0]

    ax_map.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax_map.set_ylim(-0.5, GRID_SIZE - 0.5)

    # 【Step 2: 翻转 Y 轴】 (让 Row 0 在最上面)
    ax_map.invert_yaxis()

    # 标记轴名称，防止混淆
    ax_map.set_xlabel("Column (X)")
    ax_map.set_ylabel("Row (Y)")
    ax_map.xaxis.set_ticks_position('top')  # 把 X 轴刻度放到上面更符合矩阵直觉
    ax_map.xaxis.set_label_position('top')

    ax_map.set_xticks(range(GRID_SIZE))
    ax_map.set_yticks(range(GRID_SIZE))
    ax_map.grid(True, linestyle="--", linewidth=0.5)

    cmap = plt.get_cmap("tab10")
    agent_colors = [cmap(i) for i in range(n_agents)]

    # 画障碍物
    if obstacles is not None and len(obstacles) > 0:
        obs_raw = obstacles * (GRID_SIZE - 1)
        # 同样交换
        obs_x, obs_y = obs_raw[:, 1], obs_raw[:, 0]
        ax_map.scatter(obs_x, obs_y, s=300, marker='s', color='black', alpha=1.0, edgecolors='none', zorder=1)

    # 画 Agents
    for i in range(n_agents):
        is_ego = (i == agent_id)
        c = agent_colors[i]
        marker = '*' if is_ego else 'o'
        size = 350 if is_ego else 150

        # GT
        ax_map.scatter(g_x[i], g_y[i], s=size, marker=marker, color=c, edgecolors='black', linewidth=1.5, zorder=5)
        # Belief
        ax_map.scatter(b_x[i], b_y[i], s=size, marker=marker, color=c, alpha=0.3, linewidth=0, zorder=4)
        # 连线
        dist = np.linalg.norm(g_raw[i] - b_raw[i])
        if dist > 0.5:
            ax_map.plot([g_x[i], b_x[i]], [g_y[i], b_y[i]], color=c, linestyle=':', alpha=0.6)

    # 画 Prey
    ax_map.scatter(g_x[-1], g_y[-1], s=300, marker='X', color='red', edgecolors='black', zorder=5)
    ax_map.scatter(b_x[-1], b_y[-1], s=300, marker='X', color='red', alpha=0.3, zorder=4)

    # 使用正确的 Action Name
    # 这里的 probs_orig 对应的 index 需要映射到新的名字
    title_act = ACTION_NAMES[np.argmax(probs_orig)]
    ax_map.set_title(f"{title}\n(Act: {title_act}) [Agent {agent_id} @ Step {step}]")

    # ==================== 2. 画柱状图 ====================
    x = np.arange(len(ACTION_NAMES))
    width = 0.35

    rects1 = ax_bar.bar(x - width / 2, probs_orig, width, label='Original', color='skyblue', edgecolor='black')
    if probs_new is not None:
        rects2 = ax_bar.bar(x + width / 2, probs_new, width, label='Counterfactual', color='orange', edgecolor='black')
        target_idx = np.argmax(probs_new)
        ax_bar.get_xticklabels()[target_idx].set_color('red')
        ax_bar.get_xticklabels()[target_idx].set_fontweight('bold')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ACTION_NAMES)  # 这里会自动应用正确的标签
    ax_bar.set_ylim(0, 1.1)
    ax_bar.legend()
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax_bar.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    if probs_new is not None: autolabel(rects2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- 辅助函数：从环境获取GT (保持不变) ---
def get_ground_truth_from_env(env, args):
    # (此函数在 eval 中其实用不到，因为数据里已经有了，但保留以防万一)
    pass


# --- 3. 加载模型 (保持不变) ---
def load_models():
    # ... (代码保持不变，请直接复用之前的) ...
    # 为了节省篇幅，这里省略。请确保 DualStageGenerator 类定义在上面
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default="predator_prey")
    parser.add_argument('--nagents', type=int, default=5)
    parser.add_argument('--hid_size', default=64, type=int)
    parser.add_argument('--load', default='./saved_model/lec_no_cnn_dl5_d12_r2', type=str)
    # ... 其他参数 ...
    parser.add_argument('--ic3net', action='store_true', default=False)
    parser.add_argument('--commnet', action='store_true', default=False)
    parser.add_argument('--hard_attn', action='store_true', default=False)
    parser.add_argument('--comm_mode', type=str, default='avg')
    parser.add_argument('--comm_passes', type=int, default=1)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--mean_ratio', default=1.0, type=float)
    parser.add_argument('--rnn_type', default='MLP', type=str)
    parser.add_argument('--detach_gap', default=10, type=int)
    parser.add_argument('--comm_init', default='uniform', type=str)
    parser.add_argument('--comm_action_one', default=False, action='store_true')
    parser.add_argument('--advantages_per_action', default=False, action='store_true')
    parser.add_argument('--share_weights', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
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
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--display', action="store_true", default=False)
    parser.add_argument('--random', action='store_true', default=False)

    init_args_for_env(parser)
    args, _ = parser.parse_known_args()

    # Special logic
    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0
        if args.env_name == "traffic_junction":
            args.comm_action_one = True

    args.nfriendly = args.nagents
    env = data.init(args.env_name, args, False)

    args.num_actions = env.num_actions
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.num_inputs = env.observation_dim

    if args.hard_attn and args.commnet:
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1

    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'

    parse_action_args(args)

    # Load BEPAL
    print(f"Loading BEPAL from {args.load}...")
    bepal = CommNetMLP(args, env.observation_dim)
    if os.path.exists(args.load):
        d = torch.load(args.load, map_location='cpu')
        bepal.load_state_dict(d['policy_net'])
    bepal.eval()

    # Load Route A Generator
    gen_path = "./Probe_case/generator_route_a.pt"
    print(f"Loading Generator from {gen_path}...")
    num_actions = args.num_actions[0]
    gen = DualStageGenerator(args.hid_size, num_actions)
    if os.path.exists(gen_path):
        gen.load_state_dict(torch.load(gen_path, map_location='cpu'))
    gen.eval()

    return bepal, gen, args


# --- 4. 主循环 ---
def main():
    bepal, gen, args = load_models()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bepal = bepal.to(device)
    gen = gen.to(device)

    # 加载包含 GT 和 Obstacles 的数据
    data_path = './Probe_case/data_h_with_gt.pt'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    dataset = torch.load(data_path)
    h_data = dataset['h'].to(device)
    gt_data = dataset['gt'].to(device)
    # obs_data 是 list，不能 .to(device)
    obs_data = dataset['obs']
    step_data = dataset['step'].to(device)
    agent_id_data = dataset['agent_id'].to(device)

    num_samples = 5
    print(f"\n--- Evaluating Route A with Probabilities ---")
    print(f"Images will be saved to: {OUTPUT_DIR}")

    indices = torch.randint(0, len(h_data), (num_samples,))

    for idx in indices:
        idx = idx.item()

        h = h_data[idx:idx + 1]
        gt = gt_data[idx]  # [N_entities, 4]
        obs = obs_data[idx]  # Tensor or None
        step = step_data[idx].item()
        agent_id = agent_id_data[idx].item()

        # 1. 原始预测 & 概率计算
        with torch.no_grad():
            logits = bepal.heads[0](h)
            # 【关键】计算 Softmax 概率
            probs_orig = F.softmax(logits, dim=1).cpu().numpy()[0]  # [5]
            orig_act = torch.argmax(logits, dim=1).item()

            b_orig = bepal.mapdecode(h).view(-1, 4)

        print(
            f"\nSample {idx} (Ag {agent_id}, Step {step}) | Original: {ACTION_NAMES[orig_act]} ({probs_orig[orig_act]:.2f})")

        if step < 3:
            print("  [Note] Early step warning.")

        # 2. 对抗攻击
        num_actions = args.num_actions[0]
        for target_act in range(num_actions):
            if target_act == orig_act: continue

            target_tensor = torch.tensor([target_act]).to(device)

            with torch.no_grad():
                delta, mask = gen(h, target_tensor, num_actions, hard=True)
                h_new = h + delta

                logits_new = bepal.heads[0](h_new)
                # 【关键】计算新概率
                probs_new = F.softmax(logits_new, dim=1).cpu().numpy()[0]  # [5]
                pred_act = torch.argmax(logits_new, dim=1).item()
                active_dims = mask.sum().item()

            if pred_act == target_act:
                print(f" -> Target {ACTION_NAMES[target_act]}: ✅ SUCCESS")
                print(f"    Sparsity: {int(active_dims)}")
                print(f"    Probs: {probs_new}")  # 控制台也能看到

                b_new = bepal.mapdecode(h_new).view(-1, 4)

                prefix = f"s{idx}_step{step}_ag{agent_id}_to_{target_act}"

                # 画图 1: BEFORE
                plot_comparison_with_probs(
                    b_orig, gt, obs,
                    probs_orig, None,  # 只传原始概率
                    agent_id, step,
                    f"BEFORE: Act {ACTION_NAMES[orig_act]}",
                    os.path.join(OUTPUT_DIR, f"{prefix}_BEFORE.png")
                )

                # 画图 2: AFTER
                plot_comparison_with_probs(
                    b_new, gt, obs,
                    probs_orig, probs_new,  # 传入对比概率
                    agent_id, step,
                    f"AFTER: Act {ACTION_NAMES[pred_act]}",
                    os.path.join(OUTPUT_DIR, f"{prefix}_AFTER.png")
                )
                print(f"    Saved images.")


if __name__ == "__main__":
    main()