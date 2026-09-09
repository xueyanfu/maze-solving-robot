import os
import numpy as np

# 如果本机使用 Python 自带的 Tcl/Tk 且 matplotlib 报找不到 TCL_LIBRARY，
# 可取消下面注释并把路径改成你自己的 Python 安装目录。
# python_base = r"C:\Path\To\Python"
# os.environ["TCL_LIBRARY"] = os.path.join(python_base, r"tcl\tcl8.6")
# os.environ["TK_LIBRARY"] = os.path.join(python_base, r"tcl\tk8.6")

import matplotlib
matplotlib.use('Agg')
import irsim

from utils import (
    batch_run,
    HybridController,
    virtual_light_sensor,
)

class AdvancedSLAMMapper:
    def __init__(self, width_m=10.0, height_m=10.0, resolution=0.1):
        self.resolution = resolution
        self.w_cells = int(width_m / resolution)
        self.h_cells = int(height_m / resolution)

        # 0.5: 未知区域 (Unexplored), 0.0: 自由空地 (Free), 1.0: 障碍物 (Occupied)
        self.slam_map = np.full((self.h_cells, self.w_cells), 0.5)

    def update_map(self, robot_state, lidar_scan):
        if lidar_scan is None or 'ranges' not in lidar_scan:
            return

        rx, ry, r_theta = robot_state[0, 0], robot_state[1, 0], robot_state[2, 0]
        ranges = np.array(lidar_scan['ranges'])
        angles = np.array(lidar_scan['angles']) if 'angles' in lidar_scan else np.linspace(-np.pi, np.pi, len(ranges))

        valid_mask = (ranges > 0.05) & (ranges < 5.0)
        v_ranges = ranges[valid_mask]
        v_angles = angles[valid_mask]

        rgx, rgy = int(rx / self.resolution), int(ry / self.resolution)

        for r, alpha in zip(v_ranges, v_angles):
            global_angle = r_theta + alpha
            ox = rx + r * np.cos(global_angle)
            oy = ry + r * np.sin(global_angle)

            gx, gy = int(ox / self.resolution), int(oy / self.resolution)

            if 0 <= gx < self.w_cells and 0 <= gy < self.h_cells:
                self.slam_map[gy, gx] = 1.0
                self.slam_map[rgy, rgx] = 0.0

    def find_nearest_frontier(self, robot_state):
        """
        在 SLAM 地图中寻找离机器人最近的“自由-未知”交界点（前沿点）
        """
        rx, ry = robot_state[0, 0], robot_state[1, 0]
        rgx, rgy = int(rx / self.resolution), int(ry / self.resolution)

        frontiers = []
        h, w = self.slam_map.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if self.slam_map[y, x] == 0.0:  # 自由空地
                    neighbors = [self.slam_map[y - 1, x], self.slam_map[y + 1, x],
                                 self.slam_map[y, x - 1], self.slam_map[y, x + 1]]
                    if 0.5 in neighbors:
                        frontiers.append((x, y))

        if not frontiers:
            return None  # 全图探索完毕

        frontiers = np.array(frontiers)
        distances = np.hypot(frontiers[:, 0] - rgx, frontiers[:, 1] - rgy)
        nearest_idx = np.argmin(distances)

        target_gx, target_gy = frontiers[nearest_idx]
        return target_gx * self.resolution, target_gy * self.resolution

if __name__ == "__main__":
    def my_hybrid_simulation(env):
        hidden_light_pos = env.custom_light_pos

        controller = HybridController(light_threshold=5.0)
        mapper = AdvancedSLAMMapper(width_m=10.0, height_m=10.0, resolution=0.1)

        for i in range(2000):
            robot_state = env.robot.state
            current_intensity = virtual_light_sensor(robot_state, hidden_light_pos)

            scan = env.get_lidar_scan()
            mapper.update_map(robot_state, scan)

            action = controller.compute_action(env, current_intensity, slam_mapper=mapper)
            env.step(action)
            env.render(0.01)

            if controller.mode == 'FOUND':
                print(f"=== 成功：已找到光源！小车已停止移动。当前步数: {i}, 光强: {current_intensity:.2f} ===")
                for _ in range(30):
                    env.step(np.array([[0.0], [0.0]]))
                    env.render(0.01)
                break

            if env.done():
                break
        env.end()

    batch_run(
        base_file_path="test.yaml",
        seeds=list(range(1, 100)),
        run_fn=my_hybrid_simulation,
    )
