import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import rcParams
import pdb

# 导入你分块保存的类
# 假设上面的代码分别保存为 track.py 和 tracker.py
from track import Track, TrackState, MatchMode
from tracker import Tracker

# 设置绘图字体（科研论文常用）
rcParams['font.family'] = "Times New Roman"


def main():
    # 1. 基础路径配置
    # 注意：请确保这些路径在你的机器上是正确的
    NPY_DIR = "D:/code/fiber/data/npy/"
    OBS_PATH = "D:/code/fiber/gitCode/pulseDL/Pred_intervals.npy"

    # 2. 加载观测数据 (Pred_intervals.npy)
    # 假设格式是列表或数组，每项对应一帧的检测区间 [(l, r), ...]
    try:
        all_observations = np.load(OBS_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"错误：无法找到观测文件 {OBS_PATH}")
        return

    # 3. 循环处理多个数据文件 (例如文件 1.npy 到 5.npy)
    for i in tqdm(range(1, 6), desc="Processing Files"):
        # 加载原始信号 arr (用于计算能量 e)
        # 截取 100:2800 范围
        try:
            arr = np.load(f"{NPY_DIR}{i}.npy")[:, 100:2800]
        except FileNotFoundError:
            print(f"跳过文件：{i}.npy 未找到")
            continue

        # 获取当前文件对应的观测区间 (假设每个文件对应 300 帧)
        obs_slice = all_observations[(i - 1) * 300: i * 300]

        # 初始化追踪器
        tracker = Tracker(max_age=3, left_b=(0, 300))

        # 用于记录特定 ID（如 ID 12）状态的临时列表
        id_12_states = []

        # 4. 核心追踪循环 (逐帧处理)
        for t_index, measurements in enumerate(obs_slice):

            # --- 步骤 A: 更新 (Update) ---
            # 传入当前帧观测、时间索引、以及当前帧的原始信号
            tracker.update(measurements, t_index, arr[t_index])

            # --- 步骤 B: 状态检查 (可选，例如观测 ID 12) ---
            target_state = tracker.get_track_state_by_id(12)
            if target_state is not None:
                id_12_states.append(target_state)

            # --- 步骤 C: 预测 (Predict) ---
            # 为下一帧做准备
            tracker.predict()

        # 5. 数据持久化与结果保存

        # 保存特定 ID 的追踪历史到 CSV
        if id_12_states:
            df_id12 = pd.DataFrame(id_12_states)
            df_id12.to_csv(f"id_12_file_{i}.csv", index=False)

        # 将所有已确认 (CONFIRMED) 的轨迹存入 records
        for tr in tracker.tracks:
            if tr.state == TrackState.CONFIRMED:
                tracker.records[tr.id] = tr.history

        # 6. 可视化 (调用你本地的 plot 模块)
        # 注意：这里需要你本地有 plot.py 及其中的 plot_histories 函数
        try:
            from plot import plot_histories
            plot_histories(
                tracker.records,
                savepath=f"tracks_v3_file_{i}_result.png"
            )
        except ImportError:
            print("警告：未找到 plot.py，跳过绘图步骤。")


if __name__ == "__main__":
    main()
