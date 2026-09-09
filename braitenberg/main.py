import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ------------------------------------------------------------------
# 1. Braitenberg 车辆速度逻辑定义
# 输入: SL (左传感器信号强度, 0-1), SR (右传感器信号强度, 0-1)
# 输出: VL (左轮速度), VR (右轮速度)
# ------------------------------------------------------------------

def vehicle_2a_logic(SL, SR):
    """恐惧 (FEAR): 兴奋, 同侧 (Excitatory, Same-side). 远离光源."""
    VL = SL
    VR = SR
    return VL, VR


def vehicle_2b_logic(SL, SR):
    """激进 (AGGRESSION): 兴奋, 交叉 (Excitatory, Crossed). 冲向光源."""
    VL = SR
    VR = SL
    return VL, VR


def vehicle_3a_logic(SL, SR):
    """喜爱 (LOVE): 抑制, 同侧 (Inhibitory, Same-side). 停在光源旁."""
    # 抑制通常表示为 (1 - S)
    VL = 1.0 - SL
    VR = 1.0 - SR
    return VL, VR


def vehicle_3b_logic(SL, SR):
    """探索 (EXPLORATION): 抑制, 交叉 (Inhibitory, Crossed). 绕过光源并远离."""
    k = 0.3     #光敏系数

    VL = (1.0 - SR) * k + 0.4
    VR = (1.0 - SL) * k + 0.4
    # 增加一个小常数，防止完全抑制为0，保持一点点前进动力
    return VL, VR


# ------------------------------------------------------------------
# 2. 核心模拟器类
# ------------------------------------------------------------------
class BraitenbergSimulation:
    def __init__(self, vehicle_logic_func, light_pos, start_pos, start_angle):
        self.logic = vehicle_logic_func
        self.light_pos = np.array(light_pos)
        self.robot_pos = np.array(start_pos, dtype=float)
        self.robot_angle = start_angle  # 弧度
        self.sensor_offset = 1.0  # 传感器相对于机器人的偏置距离
        self.dt = 0.1  # 时间步长
        self.history_pos = [self.robot_pos.copy()]

    def update(self):
        # 1. 传感器计算 (保持不变)
        sL_pos = self.robot_pos + np.array([self.sensor_offset * np.cos(self.robot_angle + 0.5),
                                            self.sensor_offset * np.sin(self.robot_angle + 0.5)])
        sR_pos = self.robot_pos + np.array([self.sensor_offset * np.cos(self.robot_angle - 0.5),
                                            self.sensor_offset * np.sin(self.robot_angle - 0.5)])

        # 2. 信号强度计算 (增加衰减系数，让行为更敏感)
        distL = np.linalg.norm(sL_pos - self.light_pos)
        distR = np.linalg.norm(sR_pos - self.light_pos)
        bias = 0.5
        SL = (1.0 / (distL + 1.0)) + bias
        SR = (1.0 / (distR + 1.0)) + bias

        # 逻辑调用
        VL, VR = self.logic(SL, SR)

        # 增加一个基准速度，保持它即使没光也在走
        base_exploration_speed = 0.3
        # 稍微降低转向灵敏度，让它转弯更圆润，模拟“搜索”的感觉
        sensitivity = 5.0
        dot_theta = (VR - VL) * sensitivity  # 差速产生旋转
        dot_s = (VL + VR) * 0.5 + base_exploration_speed  # 增加base_exploration_speed作为最小前进底速

        # 更新姿态
        self.robot_angle += dot_theta * self.dt
        self.robot_pos[0] += dot_s * np.cos(self.robot_angle) * self.dt
        self.robot_pos[1] += dot_s * np.sin(self.robot_angle) * self.dt

        self.history_pos.append(self.robot_pos.copy())

    def get_history_xy(self):
        history = np.array(self.history_pos)
        return history[:, 0], history[:, 1]

    @staticmethod
    def _rotation_matrix(angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])


# ------------------------------------------------------------------
# 3. 动态动画设置与运行
# ------------------------------------------------------------------
def run_dynamic_braitenberg_chart():
    # 仿真参数
    light_pos = [10, 10]
    start_pos = [0, 0]
    start_angle = np.pi / 6  # 30 度

    # 初始化 4 种车辆的模拟器
    sims = [
        BraitenbergSimulation(vehicle_2a_logic, light_pos, start_pos, start_angle),
        BraitenbergSimulation(vehicle_2b_logic, light_pos, start_pos, start_angle),
        BraitenbergSimulation(vehicle_3a_logic, light_pos, start_pos, start_angle),
        BraitenbergSimulation(vehicle_3b_logic, light_pos, start_pos, start_angle)
    ]

    titles = [
        "VEHICLE 2A: FEAR (Away from light)",
        "VEHICLE 2B: AGGRESSION (Towards light)",
        "VEHICLE 3A: LOVE (Stops near light)",
        "VEHICLE 3B: EXPLORATION (Navigates around)"
    ]

    # 创建一个 2x2 子图的画布
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()  # 扁平化以便迭代

    # 初始化图形对象
    robot_points = []
    light_points = []
    robot_history_plots = []

    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlim(-5, 25)
        ax.set_ylim(-5, 25)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)

        # 光源点 (黄色圆圈)
        light_points.append(ax.plot(light_pos[0], light_pos[1], 'yo', markersize=15, markeredgecolor='black')[0])

        # 机器人当前位置点 (红色点)
        robot_points.append(ax.plot(start_pos[0], start_pos[1], 'ro', markersize=8)[0])

        # 机器人历史轨迹线 (红色虚线)
        robot_history_plots.append(ax.plot([], [], 'r--', alpha=0.7)[0])

    plt.tight_layout()

    # --------------------------------------------------------------
    # 核心更新函数：FuncAnimation 调用
    # --------------------------------------------------------------
    def update_frame(frame_num):
        for i, sim in enumerate(sims):
            # 运行一步模拟
            sim.update()

            # 获取数据
            rx, ry = sim.robot_pos
            hx, hy = sim.get_history_xy()

            # 更新图形位置
            robot_points[i].set_data([rx], [ry])
            robot_history_plots[i].set_data(hx, hy)

        return [*robot_points, *robot_history_plots]

    # 运行动画
    ani = FuncAnimation(fig, update_frame, frames=200, interval=50, blit=True, repeat=False)

    plt.show()


# 运行仿真
if __name__ == "__main__":
    run_dynamic_braitenberg_chart()