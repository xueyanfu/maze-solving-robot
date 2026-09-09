import os
import csv
import time
import numpy as np

from test import AdvancedSLAMMapper
from utils import batch_run, HybridController, virtual_light_sensor

def headless_simulation(env):
    hidden_light_pos = env.custom_light_pos
    controller = HybridController(light_threshold=5.0)
    mapper = AdvancedSLAMMapper(width_m=10.0, height_m=10.0, resolution=0.1)

    metrics = {
        'success': False,
        'steps': 2000,
        'escape_count': 0,
        'blacklist_count': 0
    }

    for i in range(2000):
        robot_state = env.robot.state
        current_intensity = virtual_light_sensor(robot_state, hidden_light_pos)

        scan = env.get_lidar_scan()
        mapper.update_map(robot_state, scan)

        if controller.mode == 'ESCAPE' and controller.prev_intensity != current_intensity:
            metrics['escape_count'] += 1

        action = controller.compute_action(env, current_intensity, slam_mapper=mapper)
        env.step(action)

        if controller.mode == 'FOUND':
            metrics['success'] = True
            metrics['steps'] = i
            break

        if env.done():
            metrics['steps'] = i
            break

    env.end()

    metrics['blacklist_count'] = len(controller.blacklisted_frontiers)
    known_cells = np.sum(mapper.slam_map != 0.5)
    metrics['coverage'] = known_cells / mapper.slam_map.size

    return metrics


if __name__ == "__main__":
    print("开始进行多组重复实验 (共 1 组，每组 100 次)...")
    start_time = time.time()

    num_groups = 1
    seeds_list = list(range(1, 101))

    output_dir = os.path.join("results")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 将总汇总表路径指向该文件夹
    summary_csv = os.path.join(output_dir, "experiment_summary_1_groups.csv")

    with open(summary_csv, mode="w", newline="", encoding="utf-8") as f_sum:
        sum_writer = csv.writer(f_sum)
        sum_writer.writerow(["Group_ID", "Success_Rate", "Mean_Steps", "Mean_Escape", "Mean_Coverage"])

        # 3. 恢复外层循环
        for group in range(1, num_groups + 1):
            print(f"\n--- 正在运行第 {group}/{num_groups} 组实验 ---")

            results = batch_run(
                base_file_path="test.yaml",
                seeds=seeds_list,
                run_fn=headless_simulation
            )

            # 4. 动态拼接当前组的明细数据文件路径，并存入指定文件夹
            detail_csv = os.path.join(output_dir, f"group_{group}_data.csv")
            with open(detail_csv, mode="w", newline="", encoding="utf-8") as f_det:
                det_writer = csv.writer(f_det)
                det_writer.writerow(["Seed", "Success", "Steps", "Escape_Count", "Blacklist_Count", "Coverage"])
                for seed, metrics in results:
                    det_writer.writerow([seed, int(metrics['success']), metrics['steps'], metrics['escape_count'], metrics['blacklist_count'], f"{metrics['coverage']:.4f}"])

            # 计算本组的统计指标
            successes = [m['success'] for _, m in results]
            success_rate = (sum(successes) / len(successes)) * 100 if len(successes) > 0 else 0

            success_metrics = [m for _, m in results if m['success']]
            if success_metrics:
                mean_steps = np.mean([m['steps'] for m in success_metrics])
                mean_escape = np.mean([m['escape_count'] for m in success_metrics])
                mean_coverage = np.mean([m['coverage'] for m in success_metrics])
            else:
                mean_steps, mean_escape, mean_coverage = 0, 0, 0

            # 写入总表
            sum_writer.writerow([
                group,
                f"{success_rate:.2f}%",
                f"{mean_steps:.2f}",
                f"{mean_escape:.2f}",
                f"{mean_coverage:.4f}"
            ])

    elapsed_time = time.time() - start_time
    print(f"\n全部 1 组实验完成！总耗时: {elapsed_time:.2f} 秒")
    print(f"各组汇总及明细数据均已保存至文件夹: {os.path.abspath(output_dir)}")
