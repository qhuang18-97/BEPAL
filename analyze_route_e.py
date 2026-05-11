import torch
import numpy as np


def analyze_route_e_results():
    results_path = 'Probe_case/route_e_results.pt'
    data_path = 'Probe_case/data_key_h_with_gt.pt'

    print(f"Loading Route E results from {results_path}...")
    try:
        results = torch.load(results_path)
        dataset = torch.load(data_path)
    except FileNotFoundError:
        print("Files not found. Please run Route E first.")
        return

    gt_data = dataset['gt']
    agent_ids = dataset['agent_id']

    total_success = len(results)
    print(f"\nTotal Robust Causal Samples: {total_success}")

    # 统计容器
    prey_count = 0
    teammate_count = 0

    teammate_distances = []
    far_teammates = 0  # Dist > 0.4
    near_teammates = 0  # Dist < 0.2
    overlap_teammates = 0  # Dist < 0.05

    # 动作变化矩阵
    action_matrix = np.zeros((5, 5))  # row=old, col=new

    for res in results:
        idx = res['idx']
        target_ent = res['target_entity']
        orig_act = res['orig_act']
        new_act = res['new_act']

        # 记录动作变化
        action_matrix[orig_act, new_act] += 1

        # 获取原始物理距离
        current_gt = gt_data[idx]
        ego_id = agent_ids[idx].item()
        ego_pos = current_gt[ego_id, :2]

        # 你的环境里 5 通常是 Prey，0-4 是 Agents
        # 请根据你的实际 entity id 修改
        is_prey = (target_ent == 5)

        if is_prey:
            prey_count += 1
        elif target_ent != ego_id:  # 队友
            teammate_count += 1
            target_pos = current_gt[target_ent, :2]
            dist = torch.norm(ego_pos - target_pos).item()
            teammate_distances.append(dist)

            if dist > 0.4: far_teammates += 1
            if dist < 0.25: near_teammates += 1
            if dist < 0.05: overlap_teammates += 1

    # --- 打印报告 ---
    print(f"\n[1. Who Caused the Change?]")
    print(f"  Prey (Target):      {prey_count} ({prey_count / total_success:.1%})")
    print(f"  Teammates:          {teammate_count} ({teammate_count / total_success:.1%})")

    if teammate_count > 0:
        print(f"\n[2. Teammate Analysis (The Truth Test)]")
        print(f"  Mean Distance:      {np.mean(teammate_distances):.4f}")
        print(f"  -------------------------------------------")
        print(f"  Overlap (Dist~0):   {overlap_teammates} ({overlap_teammates / teammate_count:.1%})")
        print(f"  Nearby  (< 0.25):   {near_teammates} ({near_teammates / teammate_count:.1%})")
        print(f"  Far Away (> 0.4):   {far_teammates} ({far_teammates / teammate_count:.1%})")

        print(f"\n  >>> ROBUSTNESS CHECK <<<")
        if far_teammates / teammate_count < 0.1:
            print("  ✅ EXCELLENT! Far-away noise is largely filtered out.")
            print("     Route E successfully isolated local physical interactions.")
        elif far_teammates / teammate_count > 0.3:
            print("  ⚠️ WARNING: Still seeing many far-away interactions.")
            print("     Maybe the 'Far' agents actually matter? (Global coordination?)")
        else:
            print("  😐 GOOD: Far-away noise is reduced but present.")

    print(f"\n[3. Action Flip Pattern]")
    labels = ['Stay', 'Right', 'Down', 'Left', 'Up']
    # 打印最频繁的 3 种变化
    flat_indices = np.argsort(action_matrix.flatten())[::-1]
    print("  Most common causal flips:")
    for i in range(5):
        idx = flat_indices[i]
        r, c = divmod(idx, 5)
        count = action_matrix[r, c]
        if count > 0 and r != c:
            print(f"  {labels[r]} -> {labels[c]}: {int(count)} times")


if __name__ == "__main__":
    analyze_route_e_results()