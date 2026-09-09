# Maze-Solving Robot

一个面向初级公开项目的小型机器人自主导航项目，在 `irsim` 仿真环境中让差速驱动机器人在未知迷宫中自主探索并到达目标。

核心思路是把 **SLAM 占据栅格建图 + 前沿点探索 + 避障/光源趋向控制器** 组合起来，并加入 MATLAB 的 A* 全局路径规划和 Simulink 迷宫机器人模型作为对照与延伸。

## 核心逻辑

1. 用二维激光雷达扫描环境，维护一张占据栅格地图，格点状态为：
   - `0.0` 空闲
   - `0.5` 未知
   - `1.0` 障碍
2. 在地图上找出“空闲—未知”交界点作为前沿点，作为下一个探索目标。
3. `HybridController` 根据当前状态切换行为：正常探索、光源趋向、陷入死锁时避障/逃离。
4. `experiment_100.py` 批量运行多次仿真，统计成功率、平均步数与覆盖率。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `src/` | 主仿真代码：SLAM、控制器、批量实验与配置 |
| `braitenberg/` | Braitenberg 车辆行为仿真 |
| `astar/` | MATLAB A* 路径规划 |
| `simulink/` | Simulink 迷宫机器人模型 |
| `docs/` | 方法流程图（draw.io） |

## 环境与安装

- Python 3.9+（实测 3.13）
- MATLAB / Simulink（仅运行 `astar/` 与 `simulink/` 时需要）

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 批量仿真实验
cd src
python experiment_100.py

# Braitenberg 行为仿真
python ../braitenberg/main.py
```

> 结果默认输出到 `src/results/`。代码中如需 Tcl/Tk 路径，请按注释说明改成你自己的 Python 安装目录。
