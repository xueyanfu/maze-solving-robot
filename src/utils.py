import irsim
import re
import random
import yaml  # 新增依赖，用于读取 yaml 解析 world/robot 配置
import numpy as np
import math  # 确保在文件顶部导入 math
from irsim.world.map import resolve_obstacle_map


def get_occupancy_grid(world_cfg: dict):
    """
    根据 world 配置里的 obstacle_map 解析出占据栅格 (occupancy grid)。
    返回: grid (ndarray, 值范围通常 0~100, 越大越"实"), resolution (每格代表的物理长度)
    """
    obstacle_map_cfg = world_cfg.get("obstacle_map")
    width = world_cfg["width"]
    height = world_cfg["height"]

    grid = resolve_obstacle_map(
        obstacle_map=obstacle_map_cfg,
        world_width=width,
        world_height=height,
    )

    resolution = None
    if isinstance(obstacle_map_cfg, dict):
        resolution = obstacle_map_cfg.get("resolution", 0.1)

    return grid, resolution


def is_free(grid, resolution, x, y, radius=0.0, threshold=0.5):
    """
    检查以 (x, y) 为圆心、radius 为半径的机器人footprint是否落在空闲区域。
    """
    if grid is None:
        return True

    # 【核心修正 1】irsim 底层生成的地图矩阵通常采用物理系的 [x, y] 索引
    # 因此 shape 的第0维对应 x (width)，第1维对应 y (height)
    w_cells, h_cells = grid.shape

    # 向上取整确保检测圆完全包裹物理半径
    steps = max(1, math.ceil(radius / resolution)) if radius > 0 else 0

    cx = int(x / resolution)
    cy = int(y / resolution)

    for dx in range(-steps, steps + 1):
        for dy in range(-steps, steps + 1):
            # 只检测圆形范围内的格子
            if dx * dx + dy * dy > steps * steps:
                continue

            gx, gy = cx + dx, cy + dy

            # 【核心修正 2】越界检查和索引读取统一改回真正的 [gx, gy]
            if 0 <= gx < w_cells and 0 <= gy < h_cells:
                if grid[gx, gy] > threshold:
                    return False
            else:
                return False  # 出界视为不安全

    return True


def find_safe_position(
    world_cfg: dict,
    radius: float,
    init_guess=None,
    margin: float = 0.5,
    threshold: float = 0.5,
    max_tries: int = 2000,
    avoid_zones: list = None,  # 新增：需要避开的特定区域列表 [(x, y, radius), ...]
):
    grid, resolution = get_occupancy_grid(world_cfg)
    width = world_cfg["width"]
    height = world_cfg["height"]

    if init_guess is not None:
        x0, y0 = init_guess
        if is_free(grid, resolution, x0, y0, radius, threshold):
            return x0, y0

    for _ in range(max_tries):
        x = random.uniform(margin, width - margin)
        y = random.uniform(margin, height - margin)

        # 新增逻辑：检查是否与互斥区域（如已生成的小车）发生物理重叠
        conflict = False
        if avoid_zones:
            for ax, ay, ar in avoid_zones:
                # 若两点距离小于两者的半径之和加上一定的安全缓冲距离 (0.5m)，则判定为重叠
                if np.hypot(x - ax, y - ay) < (radius + ar + 0.5):
                    conflict = True
                    break

        if conflict:
            continue

        if is_free(grid, resolution, x, y, radius, threshold):
            return x, y

    raise RuntimeError("未能在给定尝试次数内找到安全的初始位置，请检查地图密度或增大 max_tries")
def update_yaml_robot_state(file_path: str, new_state: list, output_path: str = None):
    """
    修改 yaml 文件中 robot 块下第一个 state 字段的值，其余内容保持不变。
    new_state: 例如 [2.3, 4.1, 0]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    robot_start = None
    for i, line in enumerate(lines):
        if re.match(r"^robot:\s*$", line.rstrip("\n")):
            robot_start = i
            break

    if robot_start is None:
        raise ValueError("未找到 robot: 顶层字段")

    state_pattern = re.compile(r"(^\s*state:\s*)\[[^\]]*\]")
    replaced = False
    for i in range(robot_start, len(lines)):
        stripped = lines[i].rstrip("\n")
        if re.match(r"^\S+:", stripped) and i != robot_start:
            break  # 到达下一个顶层字段，robot 块结束
        match = state_pattern.match(stripped)
        if match:
            new_state_str = "[" + ", ".join(str(v) for v in new_state) + "]"
            lines[i] = state_pattern.sub(rf"\g<1>{new_state_str}", lines[i])
            replaced = True
            break

    if not replaced:
        raise ValueError("在 robot 块内未找到 state 字段")

    output_path = output_path or file_path
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"已将机器人初始 state 修改为 {new_state}")
def update_yaml_seed_content(lines, new_seed: int):
    """
    在已读入内存的 yaml 行内容中，定位 world 顶层块并替换其中的 seed 数值。
    返回替换后的新行列表。
    """
    lines = lines.copy()
    world_start = None
    world_end = len(lines)

    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if world_start is None:
            if re.match(r"^world:\s*$", stripped):
                world_start = i
                continue
        else:
            if re.match(r"^\S+:", stripped):
                world_end = i
                break

    if world_start is None:
        raise ValueError("未找到 world: 顶层字段，请检查文件内容")

    seed_pattern = re.compile(r"(^\s*seed:\s*)\d+")
    replaced = False

    for i in range(world_start, world_end):
        match = seed_pattern.match(lines[i].rstrip("\n"))
        if match:
            lines[i] = seed_pattern.sub(rf"\g<1>{new_seed}", lines[i])
            replaced = True
            break

    if not replaced:
        raise ValueError("在 world 块内未找到 seed 字段")

    return lines

def batch_run(base_file_path: str, seeds: list, run_fn=None):
    """
    批量运行：对每个 seed 生成配置并调用 irsim.make() 创建环境，然后执行仿真。

    :param base_file_path: 基础 yaml 文件路径
    :param seeds: 需要跑的 seed 列表
    :param run_fn: 自定义的仿真运行函数，接收 env 作为参数；不传则只创建 env 不执行仿真
    """
    results = []
    for seed in seeds:
        print(f"\n===== 正在运行 seed={seed} =====")
        env = run_with_seed(base_file_path, seed)

        if run_fn is not None:
            result = run_fn(env)
            results.append((seed, result))
        else:
            results.append((seed, env))

        # 如果 irsim 环境需要显式关闭/重置，可以在这里加 env.end() 或 env.reset()
        # env.end()

    return results


def run_with_seed(base_file_path: str, seed: int, temp_file_path: str = None,
                  margin: float = 0.5, threshold: float = 0.5):
    target_path = temp_file_path or f"temp_{base_file_path}"

    with open(base_file_path, "r", encoding="utf-8") as f:
        base_lines = f.readlines()

    new_lines = update_yaml_seed_content(base_lines, seed)
    with open(target_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    with open(target_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    world_cfg = cfg["world"]
    robot_cfg = cfg["robot"][0]
    robot_radius = robot_cfg["shape"].get("radius", 0.2)

    # 1. 优先生成小车位置
    safe_inflation_radius = robot_radius + 0.15
    rx, ry = find_safe_position(world_cfg, safe_inflation_radius, None, margin, threshold)

    orig_state = robot_cfg.get("state", [0, 0, 0])
    theta = orig_state[2] if len(orig_state) > 2 else 0
    update_yaml_robot_state(target_path, [round(rx, 3), round(ry, 3), theta], target_path)

    # 2. 生成光源位置
    light_radius = 0.4
    safe_light_radius = light_radius + 0.1

    # 【核心修复】：将已经确定的小车坐标及半径作为互斥区域传入
    avoid_list = [(rx, ry, safe_inflation_radius)]
    light_pos = generate_random_light_position(
        world_cfg,
        safe_light_radius,
        margin,
        threshold,
        avoid_zones=avoid_list
    )

    append_light_to_yaml(target_path, light_pos, light_radius)

    env = irsim.make(target_path)
    env.custom_light_pos = light_pos
    return env


class HybridController:
    def __init__(self, light_threshold=5.0, found_threshold=90.0):
        self.mode = 'WANDER'
        self.light_threshold = light_threshold
        self.found_threshold = found_threshold
        self.prev_intensity = 0.0
        self.search_w = 0.8
        self.best_intensity = 0.0
        self.bad_step_count = 0
        self.position_history = []
        self.escape_steps = 0
        self.current_path = None
        self.path_index = 0
        self.replan_cooldown = 0.8

        # 防止机器人长期贴墙运动
        self.wall_avoid_gain = 2.0

        # ========== 新增：探索记忆 ==========
        # 记录机器人已经走过哪些栅格，用于惩罚重复访问
        self.visited_map = None          # 与 slam_map 同尺寸的 0/1 矩阵
        self.visited_resolution = None
        self.frontier_stuck_count = 0    # 前沿导航卡死计数器

    def _update_visited_map(self, rx, ry, slam_mapper):
        """标记当前位置及周围为已访问"""
        if slam_mapper is None:
            return
        if self.visited_map is None:
            self.visited_map = np.zeros_like(slam_mapper.slam_map)
            self.visited_resolution = slam_mapper.resolution

        rgx = int(rx / self.visited_resolution)
        rgy = int(ry / self.visited_resolution)
        h, w = self.visited_map.shape
        # 以机器人为中心，标记周围 3 格半径为已访问
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                gx, gy = rgx + dx, rgy + dy
                if 0 <= gx < w and 0 <= gy < h:
                    self.visited_map[gy, gx] = 1.0

    def _get_exploration_bias(self, rx, ry, r_theta, slam_mapper):
        """
        根据已访问地图，计算一个"探索偏好角度"（机器人本体坐标系下）。
        优先选择前方未知且未访问的格子方向；如果没有，返回 None。
        """
        if self.visited_map is None or slam_mapper is None:
            return None

        res = slam_mapper.resolution
        rgx, rgy = int(rx / res), int(ry / res)
        h, w = self.visited_map.shape
        best_score = -1.0
        best_angle = None

        search_r = int(5.0 / slam_mapper.resolution)
        for dy in range(-search_r, search_r + 1):
            for dx in range(-search_r, search_r + 1):
                gx, gy = rgx + dx, rgy + dy
                if not (0 <= gx < w and 0 <= gy < h):
                    continue

                # 核心：偏好 "未知(0.5)" 且 "未访问(0)" 的格子
                is_unknown = (slam_mapper.slam_map[gy, gx] == 0.5)
                is_unvisited = (self.visited_map[gy, gx] < 0.5)

                if is_unknown and is_unvisited:
                    wx, wy = gx * res, gy * res
                    angle = np.arctan2(wy - ry, wx - rx) - r_theta
                    angle = (angle + np.pi) % (2 * np.pi) - np.pi

                    dist = np.hypot(wx - rx, wy - ry)
                    # 越近、越靠前（|angle| 小）得分越高
                    score = (1.0 / (dist + 0.1)) * max(0, np.cos(angle))
                    if score > best_score:
                        best_score = score
                        best_angle = angle

        return best_angle

    def compute_action(self, env, current_intensity, slam_mapper=None,
                     robot_radius=0.1, stop_margin=0.04):
        scan = env.get_lidar_scan()
        if scan is None or 'ranges' not in scan:
            return np.array([[0.0], [0.5]])

        robot_state = env.robot.state
        rx, ry = robot_state[0, 0], robot_state[1, 0]
        r_theta = robot_state[2, 0] if robot_state.shape[0] > 2 else 0.0

        # ========== 每帧更新已访问地图 ==========
        if slam_mapper is not None:
            self._update_visited_map(rx, ry, slam_mapper)

        # --- 死锁检测（原有逻辑保留）---
        if self.mode != 'FOUND' and self.mode != 'ESCAPE':
            self.position_history.append((rx, ry))
            if len(self.position_history) > 120:
                self.position_history.pop(0)
            if len(self.position_history) == 120:
                p_start = self.position_history[0]
                dist_moved = np.hypot(rx - p_start[0], ry - p_start[1])
                if dist_moved < 0.2:
                    print("[死锁检测] 陷入死锁，触发强制调头！")
                    self.mode = 'ESCAPE'
                    self.escape_steps = 40
                    self.position_history.clear()

        ranges = np.array(scan['ranges'])
        num_points = len(ranges)
        angles = np.array(scan['angles']) if 'angles' in scan else np.linspace(-np.pi, np.pi, num_points)

        valid_mask = ranges > 0.05
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)

        # 紧急刹车（原有逻辑保留）
        stop_x = robot_radius + stop_margin
        stop_y = robot_radius
        in_front_stop_zone = (x > 0) & (x < stop_x) & (np.abs(y) < stop_y)

        front_mask = (valid_angles > -np.pi / 9) & (valid_angles < np.pi / 9)
        front_ranges = valid_ranges[front_mask]
        front_angles = valid_angles[front_mask]

        left_mask = (valid_angles > 0) & (valid_angles < np.pi / 2)
        right_mask = (valid_angles < 0) & (valid_angles > -np.pi / 2)
        mean_left = np.mean(valid_ranges[left_mask]) if np.any(left_mask) else 0.0
        mean_right = np.mean(valid_ranges[right_mask]) if np.any(right_mask) else 0.0
        min_front_dist = np.min(front_ranges) if len(front_ranges) > 0 else 5.0

        # ESCAPE 状态（原有逻辑保留）
        if self.mode == 'ESCAPE':
            self.escape_steps -= 1
            v, w = 0.0, 1.2
            if self.escape_steps <= 0:
                self.mode = 'WANDER'
                print("[死锁解除] 调头完成，恢复 WANDER。")
            self.prev_intensity = current_intensity
            return np.array([[v], [w]])

        if np.any(in_front_stop_zone) and self.mode != 'ESCAPE':
            self.prev_intensity = current_intensity
            print("[紧急刹车] 车头正前方极近距离障碍物！")
            return np.array([[0.0], [0.8 if mean_left > mean_right else -0.8]])

        # ========== 状态机转换（原有逻辑保留）==========
        if self.mode == 'WANDER':
            if current_intensity >= self.found_threshold:
                self.mode = 'FOUND'
            elif current_intensity >= self.light_threshold:
                self.mode = 'TRACK'
                self.best_intensity = current_intensity
                print(f"--> [状态切换] 捕获光信号，启动记忆追踪")

        elif self.mode == 'TRACK':
            if current_intensity >= self.found_threshold:
                self.mode = 'FOUND'
                print(f"--> [光源找到]")
            elif current_intensity < self.light_threshold:
                self.mode = 'WANDER'
                print(f"--> [状态切换] 信号丢失，退回 WANDER")

        elif self.mode == 'FOUND':
            return np.array([[0.0], [0.0]])

        # ========== TRACK 模式（原有 A* 路径追踪保留）==========
        if self.mode == 'TRACK' and slam_mapper is not None:
            self.replan_cooldown -= 1
            if self.current_path is None or self.replan_cooldown <= 0:
                light_pos = env.custom_light_pos
                start_pt = (rx, ry, r_theta)
                goal_pt = (light_pos[0], light_pos[1])
                path = hybrid_a_star_planning(slam_mapper.slam_map, start_pt, goal_pt, slam_mapper.resolution)
                if path and len(path) > 1:
                    self.current_path = path
                    self.path_index = 0
                    self.replan_cooldown = 40
                    print(f"[规划成功] 生成 {len(path)} 个路点的避障路径")
                else:
                    print("[规划警告] 暂时未找到通路，退回 WANDER")
                    self.mode = 'WANDER'
                    self.current_path = None
                    return np.array([[0.0], [0.5]])

            if self.current_path is not None and self.path_index < len(self.current_path):
                target_x, target_y = self.current_path[self.path_index]
                dist_to_target = np.hypot(target_x - rx, target_y - ry)
                while dist_to_target < 0.6 and self.path_index < len(self.current_path) - 1:
                    self.path_index += 1
                    target_x, target_y = self.current_path[self.path_index]
                    dist_to_target = np.hypot(target_x - rx, target_y - ry)

                dx, dy = target_x - rx, target_y - ry
                target_angle = np.arctan2(dy, dx) - r_theta
                target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi

                if min_front_dist < 0.3:
                    w = 1.0 * target_angle + 0.5 * (mean_left - mean_right)
                    v = 0.15
                else:
                    w = 1.5 * target_angle
                    v = 1.0 * max(0.3, 1.0 - abs(target_angle) / np.pi)

                w = np.clip(w, -1.0, 1.0)
                self.prev_intensity = current_intensity
                return np.array([[v], [w]])

        # ========== 核心修改：WANDER 模式 ==========
        else:  # WANDER
            # --- 第一层：优先使用 SLAM 前沿点导航 ---
            if slam_mapper is not None:
                frontier = slam_mapper.find_nearest_frontier(robot_state)
                if frontier is not None:
                    fx, fy = frontier
                    dx, dy = fx - rx, fy - ry
                    dist_f = np.hypot(dx, dy)

                    # 如果前沿点很近但长时间到不了，说明被墙挡住了
                    if dist_f < 0.6:
                        self.frontier_stuck_count += 1
                    else:
                        self.frontier_stuck_count = 0

                    if self.frontier_stuck_count < 80:  # 约 4 秒
                        target_angle = np.arctan2(dy, dx) - r_theta
                        target_angle = (target_angle + np.pi) % (2 * np.pi) - np.pi

                        if min_front_dist < 0.5:
                            v = 0.3
                            w = 1.2 * (mean_left - mean_right) + 0.8 * target_angle
                        else:
                            v = 0.8 * max(0.3, 1.0 - abs(target_angle) / np.pi)
                            w = 1.5 * target_angle

                        w = np.clip(w, -1.0, 1.0)
                        self.prev_intensity = current_intensity
                        print(f"[前沿探索] 目标 ({fx:.1f}, {fy:.1f}), 距 {dist_f:.2f}m, "
                              f"角 {np.degrees(target_angle):.0f}°")
                        return np.array([[v], [w]])
                    else:
                        print("[前沿卡死] 切换随机突围")
                        self.frontier_stuck_count = 0

            # --- 第二层：退回到改进的 LIDAR 策略，加入"已知区域惩罚" ---
            explore_angle = self._get_exploration_bias(rx, ry, r_theta, slam_mapper)

            v_raw = 0.8 + 0.2 * ((min_front_dist - 0.5) / 1.0)#follow wall
            v = np.clip(v_raw, 0.6, 1.0)

            if len(front_ranges) > 0:
                max_range = np.max(front_ranges)
                max_angles = front_angles[front_ranges == max_range]
                lidar_angle = max_angles[np.argmin(np.abs(max_angles))]
            else:
                lidar_angle = 0.0
                max_range = 5.0

            # 融合：LIDAR 保证不撞墙，explore_angle 鼓励去未知区域
            if explore_angle is not None:
                # 如果探索方向在安全范围内，加大权重；否则以 LIDAR 为主
                target_angle = 0.5 * lidar_angle + 0.5 * explore_angle
                print(f"[混合探索] LIDAR: {np.degrees(lidar_angle):.0f}°, "
                      f"探索: {np.degrees(explore_angle):.0f}°")
            else:
                target_angle = lidar_angle

            # ===============================
            # 增加墙壁距离平衡控制
            # ===============================

            # 左右距离差
            wall_error = mean_left - mean_right

            # 1. 设置控制死区：微小误差不触发纠偏，防止原地高频震荡
            if abs(wall_error) < 0.15:
                wall_correction = 0.0
            else:
                # 2. 降低增益，采用线性缓和的纠偏替代硬编码符号函数
                wall_correction = -1.0 * np.clip(wall_error, -0.5, 0.5)

            if min_front_dist < 1.0:

                w = (
                    # 接近障碍物时的柔和避障
                    0.5 * wall_error + 0.5 * target_angle
                )

                print(
                    f"[游走避障] 前方{min_front_dist:.2f}m "
                    f"墙修正:{wall_correction:.2f}"
                )

            else:

                w = (
                    # 开阔或平行过墙时的平稳高速通过：降低墙差权重，提高目标导向
                    0.2 * wall_correction + 1.0 * target_angle
                )

                print(
                    f"[游走冲刺] 开阔地 {max_range:.2f}m "
                    f"墙差:{wall_error:.2f}"
                )

            w = np.clip(w, -0.6, 0.6)
            self.prev_intensity = current_intensity
            return np.array([[v], [w]])


def virtual_light_sensor(robot_state, light_pos, base_intensity=100.0):
    """
    虚拟光强传感器
    """
    rx, ry = robot_state[0, 0], robot_state[1, 0]
    dist_sq = (rx - light_pos[0]) ** 2 + (ry - light_pos[1]) ** 2
    intensity = base_intensity / (dist_sq + 1e-5)
    noise = np.random.normal(0, intensity * 0.05)
    return max(0.0, intensity + noise)

def generate_random_light_position(
    world_cfg: dict,
    light_radius: float = 0.4,
    margin: float = 0.5,
    threshold: float = 0.5,
    avoid_zones: list = None
):
    # 将互斥区域透传给底层的采样函数
    lx, ly = find_safe_position(
        world_cfg=world_cfg,
        radius=light_radius,
        init_guess=None,
        margin=margin,
        threshold=threshold,
        avoid_zones=avoid_zones
    )
    return [round(lx, 3), round(ly, 3)]

def append_light_to_yaml(file_path: str, light_pos: list, radius: float = 0.4):
    """
    向已存在的 YAML 文件末尾追加光源节点的视觉配置。
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n")  # 核心修复：强制起一个新行，防止与原文件的最后一行粘连
        f.write(f"  - shape: {{name: 'circle', radius: {radius}}}\n")
        f.write(f"    state: [{light_pos[0]:.3f}, {light_pos[1]:.3f}, 0]\n")
        f.write(f"    color: 'y'\n")


import heapq


def hybrid_a_star_planning(occupancy_map, start_node, goal_node, resolution):
    """
    基于连续状态空间和运动学约束的混合 A* (Hybrid A*) 路径规划
    occupancy_map: 0.0 为自由，1.0 为障碍物，0.5 为未知
    start_node: (x, y, theta) - 必须包含初始朝向
    goal_node: (x, y) - 终点仅作位置约束
    """
    h, w = occupancy_map.shape

    # 解析起点与终点
    sx, sy, stheta = start_node
    gx, gy = goal_node

    # 边界与终点合法性检查：如果终点在障碍物内，尝试在周围找个最近的自由点
    target_gx, target_gy = int(gx / resolution), int(gy / resolution)
    target_gx = np.clip(target_gx, 0, w - 1)
    target_gy = np.clip(target_gy, 0, h - 1)

    if occupancy_map[target_gy, target_gx] >= 1.0:
        found_free = False
        for r_offset in range(1, 10):
            for dx in range(-r_offset, r_offset + 1):
                for dy in range(-r_offset, r_offset + 1):
                    nx, ny = target_gx + dx, target_gy + dy
                    if 0 <= nx < w and 0 <= ny < h and occupancy_map[ny, nx] < 1.0:
                        gx, gy = nx * resolution, ny * resolution
                        found_free = True
                        break
                if found_free: break
            if found_free: break

    # 1. 定义运动学原语 (v, w)
    # v: 线速度 (m/s), w: 角速度 (rad/s), dt: 单步积分时间 (s)
    dt = 0.4
    primitives = [
        (0.5, 0.0),  # 直行
        (0.5, 0.8),  # 左前转弯
        (0.5, -0.8),  # 右前转弯
        (-0.3, 0.0),  # 后退直行
        (-0.3, 0.8),  # 左后转弯
        (-0.3, -0.8)  # 右后转弯
    ]

    # 2. 状态离散化参数（用于 Closed Set 查重，防止连续空间无限展开）
    theta_res = np.deg2rad(15)  # 角度以 15 度为分辨率离散化

    def get_grid_idx(x, y, theta):
        return (int(x / resolution), int(y / resolution), int((theta % (2 * np.pi)) / theta_res))

    start_idx = get_grid_idx(sx, sy, stheta)

    # 优先队列: (f_score, g_score, (x, y, theta), path_points)
    open_set = []
    heapq.heappush(open_set, (0.0, 0.0, (sx, sy, stheta), [(sx, sy)]))

    visited = set()
    visited.add(start_idx)

    # 辅助碰撞检测（以小车视为质点加少量安全余量，可通过膨胀地图优化）
    def is_free(x, y):
        grid_x, grid_y = int(x / resolution), int(y / resolution)
        if 0 <= grid_x < w and 0 <= grid_y < h:
            return occupancy_map[grid_y, grid_x] < 1.0
        return False

    while open_set:
        f, g, (cx, cy, ctheta), path = heapq.heappop(open_set)

        # 抵达判定：考虑到连续状态无法精确命中终点坐标，设置 0.3m 的容差半径
        if np.hypot(cx - gx, cy - gy) < 0.3:
            return path

        for v, w_vel in primitives:
            valid = True
            nx, ny, ntheta = cx, cy, ctheta
            temp_path = []

            # 将一个长动作切分为多次小步幅积分与碰撞检测
            steps = 4
            step_dt = dt / steps

            for _ in range(steps):
                nx += v * np.cos(ntheta) * step_dt
                ny += v * np.sin(ntheta) * step_dt
                ntheta += w_vel * step_dt
                temp_path.append((nx, ny))

                if not is_free(nx, ny):
                    valid = False
                    break

            if not valid:
                continue

            ntheta = ntheta % (2 * np.pi)
            n_idx = get_grid_idx(nx, ny, ntheta)

            if n_idx not in visited:
                visited.add(n_idx)

                # 3. 代价计算
                distance_cost = abs(v * dt)
                turn_penalty = 1.2 * abs(w_vel * dt)  # 转弯惩罚，促使路径尽可能走直线
                reverse_penalty = 2.0 if v < 0 else 0.0  # 强烈惩罚后退操作

                new_g = g + distance_cost + turn_penalty + reverse_penalty

                # 启发式: 简单的欧氏距离
                h_val = np.hypot(gx - nx, gy - ny)
                new_f = new_g + h_val

                heapq.heappush(open_set, (new_f, new_g, (nx, ny, ntheta), path + temp_path))

    return None
class MultiLightSensorArray:
    """
    多方位光敏电阻阵列
    - 传感器围绕机器人中心呈圆形分布
    - 支持距离衰减、方向性余弦权重、障碍物射线遮挡检测
    """
    def __init__(self, num_sensors=8, radius=0.15, fov_deg=120.0,
                 base_intensity=100.0, attenuation_k=1.0):
        """
        num_sensors: 传感器数量（默认8个，均匀分布在360°）
        radius: 传感器距离机器人中心的安装半径（米）
        fov_deg: 每个传感器的视场角（度），越窄指向性越强
        base_intensity: 光源基准强度
        attenuation_k: 距离衰减系数，I = I0 / (1 + k * d^2)
        """
        self.num_sensors = num_sensors
        self.radius = radius
        self.fov = np.deg2rad(fov_deg)
        self.base_intensity = base_intensity
        self.attenuation_k = attenuation_k

        # 均匀分布在机器人本体坐标系下，alpha_i 为相对安装角
        self.alphas = np.linspace(0, 2 * np.pi, num_sensors, endpoint=False)

    def get_sensor_global_positions(self, robot_state):
        """
        根据机器人位姿计算各传感器在全局地图中的坐标
        robot_state: [[x], [y], [theta]] 或兼容形状
        """
        rx = robot_state[0, 0]
        ry = robot_state[1, 0]
        rtheta = robot_state[2, 0] if robot_state.shape[0] > 2 else 0.0

        positions = []
        for alpha in self.alphas:
            sx = rx + self.radius * np.cos(rtheta + alpha)
            sy = ry + self.radius * np.sin(rtheta + alpha)
            positions.append((sx, sy, alpha))
        return positions

    def raycast_blocked(self, start_xy, end_xy, occupancy_map, resolution):
        """
        射线遮挡检测：从传感器位置到光源之间是否有障碍物
        occupancy_map: 0.0=自由, 1.0=障碍, 0.5=未知（与 AdvancedSLAMMapper 格式一致）
        返回: True=被遮挡, False=无遮挡
        """
        if occupancy_map is None:
            return False

        x0, y0 = start_xy
        x1, y1 = end_xy
        dx = x1 - x0
        dy = y1 - y0
        dist = np.hypot(dx, dy)
        if dist < 1e-6:
            return False

        # 沿射线均匀采样，步长不超过半个栅格
        steps = max(int(dist / (resolution * 0.5)), 10)
        h, w = occupancy_map.shape

        for i in range(steps + 1):
            t = i / steps
            x = x0 + t * dx
            y = y0 + t * dy
            gx = int(x / resolution)
            gy = int(y / resolution)

            if 0 <= gx < w and 0 <= gy < h:
                if occupancy_map[gy, gx] >= 1.0:  # 障碍物
                    return True
            else:
                # 超出地图边界视为遮挡（安全保守策略）
                return True
        return False

    def read(self, robot_state, light_pos, occupancy_map=None, resolution=0.1):
        """
        读取阵列中所有传感器的光强值
        返回: list[dict]，每个元素包含 sensor_id, angle, position, intensity, blocked 等
        """
        sensor_positions = self.get_sensor_global_positions(robot_state)
        readings = []

        for i, (sx, sy, alpha) in enumerate(sensor_positions):
            # 1. 距离衰减（反比平方定律）
            dx = light_pos[0] - sx
            dy = light_pos[1] - sy
            dist_sq = dx**2 + dy**2
            I_base = self.base_intensity / (1.0 + self.attenuation_k * dist_sq)

            # 2. 方向性权重（视角惩罚）
            # 传感器法线方向 = 机器人朝向 + 相对安装角
            rx = robot_state[0, 0]
            ry = robot_state[1, 0]
            rtheta = robot_state[2, 0] if robot_state.shape[0] > 2 else 0.0
            sensor_normal = rtheta + alpha
            light_dir = np.arctan2(dy, dx)
            angle_diff = light_dir - sensor_normal
            # 归一化到 [-pi, pi]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            half_fov = self.fov / 2.0
            if abs(angle_diff) > half_fov:
                directional_weight = 0.0
            else:
                # 余弦衰减：正对光源时 cos(0)=1，边缘处衰减至 0
                directional_weight = max(0.0, np.cos(angle_diff))

            # 3. 障碍物遮挡检测（可选，需传入 occupancy_map）
            blocked = False
            if occupancy_map is not None:
                blocked = self.raycast_blocked(
                    (sx, sy), light_pos, occupancy_map, resolution
                )

            # 4. 合成最终光强
            if blocked:
                intensity = 0.0
            else:
                intensity = I_base * directional_weight

            # 5. 加入传感器噪声（5% 相对噪声）
            if intensity > 1e-6:
                noise = np.random.normal(0, intensity * 0.05)
                intensity = max(0.0, intensity + noise)

            readings.append({
                'sensor_id': i,
                'alpha': alpha,                     # 本体坐标系下的相对角
                'global_position': (sx, sy),        # 全局坐标
                'intensity': intensity,             # 最终光强读数
                'blocked': blocked,                 # 是否被遮挡
                'directional_weight': directional_weight,
                'distance': np.sqrt(dist_sq)
            })

        return readings

    def estimate_light_direction(self, readings):
        """
        根据阵列读数加权估计光源方向（相对于机器人本体）
        返回: estimated_angle（弧度，以机器人正前方为0，左正右负）
               若所有读数都为0，返回 None
        """
        total_weight = 0.0
        angle_sin = 0.0
        angle_cos = 0.0

        for r in readings:
            w = r['intensity']
            if w < 1e-6:
                continue
            total_weight += w
            angle_sin += w * np.sin(r['alpha'])
            angle_cos += w * np.cos(r['alpha'])

        if total_weight < 1e-6:
            return None

        # 加权平均方向
        avg_alpha = np.arctan2(angle_sin / total_weight, angle_cos / total_weight)
        return avg_alpha

    def get_max_intensity(self, readings):
        """获取阵列中的最大光强读数（常用于状态机阈值判断）"""
        return max(r['intensity'] for r in readings) if readings else 0.0