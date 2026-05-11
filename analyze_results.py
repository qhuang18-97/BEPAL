import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze_route_c():
    path = 'Probe_case/route_c_results.pt'
    print(f"Loading results from {path}...")
    results = torch.load(path)

    if len(results) == 0:
        print("No successful attacks recorded.")
        return

    print(f"Total Successful Samples: {len(results)}")

    # 1. 实体分布统计
    target_entities = [r['target_entity'] for r in results]
    counts = np.bincount(target_entities)
    print("\n--- Target Entity Distribution ---")
    # 假设 0-4 是 Agents, 5 是 Prey, 6+ 是 Obstacles/Padding
    for i, count in enumerate(counts):
        if count > 0:
            role = "Agent" if i < 5 else ("Prey" if i == 5 else "Obstacle/Other")
            print(f"Entity {i} ({role}): {count} times ({count / len(results):.1%})")

    # 2. 扰动幅度统计
    deltas = np.array([np.linalg.norm(r['delta'][:2]) for r in results])  # 只看 XY 位移
    print("\n--- Perturbation Magnitude (XY) ---")
    print(f"Mean Delta: {np.mean(deltas):.4f}")
    print(f"Max Delta:  {np.max(deltas):.4f}")
    print(f"Min Delta:  {np.min(deltas):.4f}")

    # 3. 动作变化统计
    print("\n--- Action Flip Examples (First 5) ---")
    action_dict = {0: 'Stay', 1: 'Right', 2: 'Down', 3: 'Left', 4: 'Up'}  # 假设的动作映射
    for i in range(min(5, len(results))):
        r = results[i]
        d = r['delta']
        print(
            f"Sample {r['idx']}: {action_dict.get(r['orig_act'], str(r['orig_act']))} -> {action_dict.get(r['new_act'], str(r['new_act']))}")
        print(f"   Moved Ent {r['target_entity']} by dx={d[0]:.2f}, dy={d[1]:.2f}")


import torch
import numpy as np


def analyze_robust_causality():
    path = 'Probe_case/route_c_macro_results.pt'
    print(f"Loading results from {path}...")
    results = torch.load(path)

    # --- 1. 定义“显著”门槛 ---
    # 门槛 1: 必须真的动了 (排除 Adversarial Hack)
    # 门槛 2: 必须动得够大 (排除 Jitter/Noise)
    # 0.05 大约是半个格子 (12x12 grid)
    THRESHOLD = 0.1

    robust_results = []

    print(f"\n[Filtering Criteria] Delta Norm >= {THRESHOLD:.4f}")

    for r in results:
        delta_mag = np.linalg.norm(r['delta'][:2])  # 只看位置变化
        if delta_mag >= THRESHOLD:
            r['mag'] = delta_mag
            robust_results.append(r)

    print(f"Original Success: {len(results)}")
    print(f"Robust Success:   {len(robust_results)} (Drop rate: {1 - len(robust_results) / len(results):.1%})")

    if len(robust_results) == 0:
        print("No robust causal samples found! Your policy is likely relying on micro-features.")
        return

    # --- 2. 重新分析“真实”因果 ---
    # 实体分布
    target_entities = [r['target_entity'] for r in robust_results]
    counts = np.bincount(target_entities, minlength=7)

    print("\n--- ROBUST Target Entity Distribution ---")
    total = len(robust_results)
    for i, count in enumerate(counts):
        if count > 0:
            role = "Agent" if i < 5 else ("Prey" if i == 5 else "Obstacle")
            print(f"Entity {i} ({role}): {count} ({count / total:.1%})")

    # 宏观变化示例
    print("\n--- Macro-Causal Examples ---")
    action_dict = {0: 'Stay', 1: 'Right', 2: 'Down', 3: 'Left', 4: 'Up'}
    for i in range(min(5, len(robust_results))):
        r = robust_results[i]
        d = r['delta']
        print(
            f"Sample {r['idx']}: {action_dict.get(r['orig_act'], str(r['orig_act']))} -> {action_dict.get(r['new_act'], str(r['new_act']))}")
        print(f"   Target: Entity {r['target_entity']} | Moved by {r['mag']:.4f} (dx={d[0]:.2f}, dy={d[1]:.2f})")


import torch
import numpy as np


def analyze_overlap_and_distance():
    # 1. 加载结果
    results_path = 'Probe_case/route_c_macro_results.pt'
    data_path = 'Probe_case/data_key_h_with_gt.pt'

    print(f"Loading results from {results_path}...")
    try:
        results = torch.load(results_path)
        dataset = torch.load(data_path)
    except FileNotFoundError:
        print("Files not found.")
        return

    gt_data = dataset['gt']
    agent_ids = dataset['agent_id']

    # 统计容器
    distances = []
    overlap_count = 0  # 完全重叠 (Dist < 0.01)
    nearby_count = 0  # 紧邻 (Dist < 0.2)
    far_count = 0  # 远端 (Dist > 0.4)

    total_teammates_moved = 0

    print(f"\nAnalyzing {len(results)} successful samples...")

    for res in results:
        idx = res['idx']
        target_ent = res['target_entity']
        ego_agent_id = agent_ids[idx].item()

        # 只分析队友 (Entity 0-4) 且不是自己
        if target_ent < 5 and target_ent != ego_agent_id:
            total_teammates_moved += 1

            # 获取坐标
            current_gt = gt_data[idx]
            ego_pos = current_gt[ego_agent_id, :2]
            target_pos = current_gt[target_ent, :2]

            # 计算距离
            dist = torch.norm(ego_pos - target_pos).item()
            distances.append(dist)

            # 分类统计
            if dist < 0.01:  # 几乎重叠 (考虑浮点误差)
                overlap_count += 1
            elif dist < 0.25:  # 约 2-3 个格子内
                nearby_count += 1
            elif dist > 0.4:  # 远端
                far_count += 1

    # --- 结果判决 ---
    print(f"\n[Teammate Interaction Analysis]")
    print(f"Total Teammate-Targeted Samples: {total_teammates_moved}")

    if total_teammates_moved == 0:
        print("No teammates were targeted. Analysis skipped.")
        return

    print(f"\n1. Overlapping Teammates (Dist ~= 0): {overlap_count} ({overlap_count / total_teammates_moved:.1%})")
    print(f"2. Nearby Teammates (0 < Dist < 0.25):  {nearby_count} ({nearby_count / total_teammates_moved:.1%})")
    print(f"3. Far Teammates (Dist > 0.4):          {far_count} ({far_count / total_teammates_moved:.1%})")

    print(f"\nMean Distance: {np.mean(distances):.4f}")

    print("\n[CONCLUSION]")
    if overlap_count / total_teammates_moved > 0.4:
        print("✅ GOOD: The policy is learning 'De-confliction/Coverage'.")
        print("   It reacts to overlapping teammates to optimize efficiency.")
    elif far_count / total_teammates_moved > 0.3:
        print("⚠️ BAD: The policy is suffering from 'Non-Local Robustness Failure'.")
        print("   It reacts to teammates that are too far away to matter.")
    else:
        print("😐 NEUTRAL: It's reacting to local density (crowding), possibly for spacing.")


import torch
import numpy as np


def inspect_perturbation_details():
    results_path = 'Probe_case/route_c_macro_results.pt'
    data_path = 'Probe_case/data_key_h_with_gt.pt'

    print(f"Loading data...")
    results = torch.load(results_path)
    dataset = torch.load(data_path)

    gt_data = dataset['gt']
    agent_ids = dataset['agent_id']

    print("\n--- Deep Dive into Overlap Perturbations (Dist < 0.05) ---")

    action_dict = {0: 'Stay', 1: 'Right', 2: 'Down', 3: 'Left', 4: 'Up'}
    count = 0

    for res in results:
        idx = res['idx']
        target_ent = res['target_entity']
        ego_id = agent_ids[idx].item()

        # 1. 基础信息获取
        current_gt = gt_data[idx]
        ego_pos = current_gt[ego_id, :2]
        target_pos = current_gt[target_ent, :2]
        prey_pos = current_gt[5, :2]  # 假设 5 是 Prey

        # 计算原始距离
        agent_dist = torch.norm(ego_pos - target_pos).item()
        dist_to_prey = torch.norm(ego_pos - prey_pos).item()

        # 只看重叠案例 (Dist < 0.05) 且 针对队友
        if agent_dist < 0.05 and target_ent < 5 and target_ent != ego_id:
            count += 1

            # 2. 获取扰动信息
            delta = res['delta']  # numpy array
            delta_xy = delta[:2]

            # 3. 计算修改后的 Belief 位置
            # 注意：这里是简单的线性叠加，模拟 Decoder 看到的画面
            # 我们需要把 tensor 转 numpy 方便计算
            orig_pos_np = target_pos.numpy()
            modified_pos_np = orig_pos_np + delta_xy

            # 物理约束 Clamp (0-1)
            modified_pos_clamped = np.clip(modified_pos_np, 0.0, 1.0)

            # 4. 打印详情
            print(f"\n[Case {count}] Sample {idx}")
            print(f"  Scenerio:     Ego & Teammate overlapping at {orig_pos_np}")
            print(f"                Distance to Prey: {dist_to_prey:.4f}")

            print(
                f"  Action Flip:  {action_dict.get(res['orig_act'], str(res['orig_act']))} -> {action_dict.get(res['new_act'], str(res['new_act']))}")

            print(f"  PERTURBATION:")
            print(f"    Vector:     dx={delta_xy[0]:.2f}, dy={delta_xy[1]:.2f}")
            print(f"    Magnitude:  {np.linalg.norm(delta_xy):.4f} (Target was ~0.125)")

            print(f"  RESULT:")
            print(f"    Old Pos:    {orig_pos_np}")
            print(f"    New Pos:    {modified_pos_clamped}")

            # 分析移动方向相对于 Ego 的关系
            # 它是把队友“推远”了吗？
            new_dist = np.linalg.norm(ego_pos.numpy() - modified_pos_clamped)
            print(f"    Effect:     Teammate pushed away! (Dist: {agent_dist:.2f} -> {new_dist:.2f})")

            if count >= 5:  # 只看前 5 个典型的
                break


if __name__ == "__main__":
    analyze_overlap_and_distance()