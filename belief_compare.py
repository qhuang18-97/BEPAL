import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ====== 配置区域 ======
BUFFER_DIR = "belief_buffer"                # .pt 所在目录
INPUT_FILE = "belief_sample_00000.pt"       # 目前你已有的文件名
OUTPUT_DIR = "belief_viz"                   # 输出图片根目录
GRID_SIZE = 12                              # = env dim, 比如 12
OFFSET_SCALE_FACTOR = 5.0                   # = trainer 里 scale_factor

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_belief_step(node_decoded, node_gt, t, agent_id, grid_size, save_path=None):
    """
    绘制某时间步 t、某个 agent_id 的 belief 和 GT。
    每个 agent 使用不同颜色；prey 单独颜色。
    第 2–3 维作为 offset，在图上画成箭头（GT 深色、belief 浅色）。
    legend 放在图外右侧。
    """
    pred_feat = node_decoded[t, agent_id]  # [N+1, F]
    gt_feat   = node_gt[t, agent_id]       # [N+1, F]

    # ---- 位置：前 2 维 ----
    gt_xy_norm   = gt_feat[:, :2]
    pred_xy_norm = pred_feat[:, :2]

    gt_xy   = gt_xy_norm * (grid_size - 1)
    pred_xy = pred_xy_norm * (grid_size - 1)

    # ---- offset：第 2–3 维（如果有的话）----
    has_offset = gt_feat.shape[1] >= 4
    if has_offset:
        gt_off_norm   = gt_feat[:, 2:4]
        pred_off_norm = pred_feat[:, 2:4]

        # 把 offset 也放大到网格尺度，作为位移向量
        gt_off   = gt_off_norm * (grid_size - 1)
        pred_off = pred_off_norm * (grid_size - 1)
    else:
        gt_off = pred_off = None

    n_entities = gt_xy.shape[0]
    n_agents   = n_entities - 1  # 最后一个是 prey

    agents_gt_xy   = gt_xy[:n_agents]
    prey_gt_xy     = gt_xy[n_agents]
    agents_pred_xy = pred_xy[:n_agents]
    prey_pred_xy   = pred_xy[n_agents]

    if has_offset:
        agents_gt_off   = gt_off[:n_agents]
        prey_gt_off     = gt_off[n_agents]
        agents_pred_off = pred_off[:n_agents]
        prey_pred_off   = pred_off[n_agents]

    # -------- colors --------
    cmap = plt.cm.get_cmap("tab10", n_agents)
    agent_colors = [cmap(i) for i in range(n_agents)]

    prey_gt_color   = "red"
    prey_pred_color = (1.0, 0.4, 0.4)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, linestyle="--", linewidth=0.5)

    # ===== collect legend entries =====
    legend_entries = []

    # -------- draw agents (点 + offset 箭头) --------
    for i in range(n_agents):
        # GT 点（深色）
        p1 = ax.scatter(
            agents_gt_xy[i, 0],
            agents_gt_xy[i, 1],
            s=85,
            marker="o",
            color=agent_colors[i],
            edgecolors="black",
            linewidths=0.7,
        )
        # belief 点（浅色）
        p2 = ax.scatter(
            agents_pred_xy[i, 0],
            agents_pred_xy[i, 1],
            s=85,
            marker="o",
            color=agent_colors[i],
            alpha=0.25,
        )

        legend_entries.append((p1, f"Agent {i} GT"))
        legend_entries.append((p2, f"Agent {i} belief"))

        # offset 箭头（如果有）
        if has_offset:
            # GT offset：深色箭头
            ax.arrow(
                agents_gt_xy[i, 0],
                agents_gt_xy[i, 1],
                agents_gt_off[i, 0],
                agents_gt_off[i, 1],
                length_includes_head=True,
                head_width=0.25,
                head_length=0.35,
                linewidth=1.0,
                color=agent_colors[i],
                alpha=0.9,
            )
            # belief offset：浅色箭头
            ax.arrow(
                agents_pred_xy[i, 0],
                agents_pred_xy[i, 1],
                agents_pred_off[i, 0],
                agents_pred_off[i, 1],
                length_includes_head=True,
                head_width=0.25,
                head_length=0.35,
                linewidth=1.0,
                color=agent_colors[i],
                alpha=0.3,
            )

    # -------- draw prey (点 + offset 箭头) --------
    p_prey_gt = ax.scatter(
        prey_gt_xy[0],
        prey_gt_xy[1],
        s=130,
        marker="X",
        color=prey_gt_color,
        edgecolors="black",
        linewidths=0.8,
    )
    p_prey_pred = ax.scatter(
        prey_pred_xy[0],
        prey_pred_xy[1],
        s=130,
        marker="X",
        color=prey_pred_color,
        alpha=0.3,
    )

    legend_entries.append((p_prey_gt, "Prey GT"))
    legend_entries.append((p_prey_pred, "Prey belief"))

    if has_offset:
        # prey 的 offset 箭头
        ax.arrow(
            prey_gt_xy[0],
            prey_gt_xy[1],
            prey_gt_off[0],
            prey_gt_off[1],
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            linewidth=1.2,
            color=prey_gt_color,
            alpha=0.9,
        )
        ax.arrow(
            prey_pred_xy[0],
            prey_pred_xy[1],
            prey_pred_off[0],
            prey_pred_off[1],
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            linewidth=1.2,
            color=prey_pred_color,
            alpha=0.3,
        )

    ax.set_title(f"t = {t}, agent = {agent_id}")

    # ===== legend outside figure =====
    handles = [item[0] for item in legend_entries]
    labels  = [item[1] for item in legend_entries]

    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=False,
    )

    # 为右侧 legend 腾点空间
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_belief_step_agent(node_decoded_agent,
                           node_gt_agent,
                           t,
                           agent_id,
                           grid_size,
                           offset_scale_factor,
                           save_path=None):
    """
    node_decoded_agent, node_gt_agent: [T, N+1, F]
    t: 时间步
    agent_id: 只是写在 title 里方便你记是谁的视角
    """
    # 当前 step 的特征 [N+1, F]
    pred_feat = node_decoded_agent[t]   # [N+1, F]
    gt_feat   = node_gt_agent[t]        # [N+1, F]

    # --- 位置：前 2 维，原来是 / (dim-1) ---
    gt_xy_norm   = gt_feat[:, :2]
    pred_xy_norm = pred_feat[:, :2]

    gt_xy   = gt_xy_norm * (grid_size - 1)
    pred_xy = pred_xy_norm * (grid_size - 1)

    # --- offset：第 2–3 维（如果存在） ---
    has_offset = gt_feat.shape[1] >= 4
    if has_offset:
        gt_off_norm   = gt_feat[:, 2:4]
        pred_off_norm = pred_feat[:, 2:4]
        # 还原到“真实位移”：乘回 (dim-1) 和 scale_factor
        gt_off   = gt_off_norm * (grid_size - 1) * offset_scale_factor
        pred_off = pred_off_norm * (grid_size - 1) * offset_scale_factor
    else:
        gt_off = pred_off = None

    n_entities = gt_xy.shape[0]
    n_agents   = n_entities - 1          # 最后一个是 prey

    agents_gt_xy   = gt_xy[:n_agents]
    agents_pred_xy = pred_xy[:n_agents]
    prey_gt_xy     = gt_xy[n_agents]
    prey_pred_xy   = pred_xy[n_agents]

    if has_offset:
        agents_gt_off   = gt_off[:n_agents]
        agents_pred_off = pred_off[:n_agents]
        prey_gt_off     = gt_off[n_agents]
        prey_pred_off   = pred_off[n_entities-1]

    # 颜色
    cmap = plt.cm.get_cmap("tab10", n_agents)
    agent_colors = [cmap(i) for i in range(n_agents)]
    prey_gt_color   = "red"
    prey_pred_color = (1.0, 0.4, 0.4)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True, linestyle="--", linewidth=0.5)

    legend_entries = []

    # ===== 画 agents（点 + 箭头） =====
    for i in range(n_agents):
        # GT 点（深色）
        p1 = ax.scatter(
            agents_gt_xy[i, 0],
            agents_gt_xy[i, 1],
            s=85,
            marker="o",
            color=agent_colors[i],
            edgecolors="black",
            linewidths=0.7,
        )
        # belief 点（浅色）
        p2 = ax.scatter(
            agents_pred_xy[i, 0],
            agents_pred_xy[i, 1],
            s=85,
            marker="o",
            color=agent_colors[i],
            alpha=0.25,
        )

        legend_entries.append((p1, f"Agent {i} GT"))
        legend_entries.append((p2, f"Agent {i} belief"))

        # offset 箭头
        if has_offset:
            ax.arrow(
                agents_gt_xy[i, 0],
                agents_gt_xy[i, 1],
                agents_gt_off[i, 0],
                agents_gt_off[i, 1],
                length_includes_head=True,
                head_width=0.25,
                head_length=0.35,
                linewidth=1.0,
                color=agent_colors[i],
                alpha=0.9,
            )
            ax.arrow(
                agents_pred_xy[i, 0],
                agents_pred_xy[i, 1],
                agents_pred_off[i, 0],
                agents_pred_off[i, 1],
                length_includes_head=True,
                head_width=0.25,
                head_length=0.35,
                linewidth=1.0,
                color=agent_colors[i],
                alpha=0.3,
            )

    # ===== 画 prey（点 + 箭头） =====
    p_prey_gt = ax.scatter(
        prey_gt_xy[0],
        prey_gt_xy[1],
        s=130,
        marker="X",
        color=prey_gt_color,
        edgecolors="black",
        linewidths=0.8,
    )
    p_prey_pred = ax.scatter(
        prey_pred_xy[0],
        prey_pred_xy[1],
        s=130,
        marker="X",
        color=prey_pred_color,
        alpha=0.3,
    )

    legend_entries.append((p_prey_gt, "Prey GT"))
    legend_entries.append((p_prey_pred, "Prey belief"))

    if has_offset:
        ax.arrow(
            prey_gt_xy[0],
            prey_gt_xy[1],
            prey_gt_off[0],
            prey_gt_off[1],
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            linewidth=1.2,
            color=prey_gt_color,
            alpha=0.9,
        )
        ax.arrow(
            prey_pred_xy[0],
            prey_pred_xy[1],
            prey_pred_off[0],
            prey_pred_off[1],
            length_includes_head=True,
            head_width=0.3,
            head_length=0.4,
            linewidth=1.2,
            color=prey_pred_color,
            alpha=0.3,
        )

    ax.set_title(f"viewer agent = {agent_id}, t = {t}")

    # legend 放图外右边
    handles = [h for h, _ in legend_entries]
    labels  = [l for _, l in legend_entries]
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    # 读取你现有的 belief_sample_00000.pt
    path = os.path.join(BUFFER_DIR, INPUT_FILE)
    data = torch.load(path)

    # [T, N, N+1, F]
    node_decoded = data["node_decoded"].numpy()
    node_gt      = data["node_gt"].numpy()

    T, N, NE, F = node_decoded.shape
    print(f"Loaded {INPUT_FILE}: T={T}, N={N}, entities={NE}, F={F}")

    for agent_id in range(N):
        # 这个 viewing agent 的 belief: [T, N+1, F]
        node_decoded_agent = node_decoded[:, agent_id]
        node_gt_agent      = node_gt[:, agent_id]

        # 为这个 agent 建一个文件夹
        out_agent_dir = os.path.join(OUTPUT_DIR, f"agent_{agent_id}")
        os.makedirs(out_agent_dir, exist_ok=True)

        for t in range(T):
            save_path = os.path.join(
                out_agent_dir,
                f"{os.path.splitext(INPUT_FILE)[0]}_t{t:03d}.png"
            )
            plot_belief_step_agent(
                node_decoded_agent=node_decoded_agent,
                node_gt_agent=node_gt_agent,
                t=t,
                agent_id=agent_id,
                grid_size=GRID_SIZE,
                offset_scale_factor=OFFSET_SCALE_FACTOR,
                save_path=save_path,
            )
            print(f"Saved {save_path}")



if __name__ == "__main__":
    main()
