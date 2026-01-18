import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = "Times New Roman"


def plot_histories(records, savepath, figsize=(8, 20)):
    """
    records: tracker.records
             {track_id: [[c, w, time], ...]}
    """
    plt.figure(figsize=figsize)

    cmap = plt.get_cmap("tab10")
    track_ids = list(records.keys())

    for k, track_id in enumerate(track_ids):
        hist = records[track_id]
        if not hist:
            continue

        color = cmap(k % cmap.N)

        # 按时间排序（保险起见）
        hist = sorted(hist, key=lambda x: x[2])

        centers = [h[0] for h in hist]
        times = [h[2] for h in hist]

        # 1️⃣ 连线（轨迹）
        plt.plot(
            centers,
            times,
            color=color,
            linewidth=1.5,
            label=f"Track {track_id}"
        )

        # 2️⃣ 中心点
        plt.scatter(
            centers,
            times,
            color=color,
            s=15,
            zorder=3
        )

    plt.xlabel("Position", fontsize=24)
    plt.ylabel("Time", fontsize=24)
    plt.title("Track center trajectories (time vs position)", fontsize=30)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    # 轨迹多时不建议开 legend
    # plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_observations_with_centers(Observations, save_path="obs_intervals_centers.png"):
    """
    Observations: list / ndarray
        Observations[t] = [[l1, r1], [l2, r2], ...]
    """

    plt.figure(figsize=(8, 20))
    t_start = 900
    c_min = 0
    c_max = 2700
    for dt, obs in enumerate(Observations):
        if obs is None or len(obs) == 0:
            continue

        t = t_start + dt

        for l, r in obs:
            c = 0.5 * (l + r)

            # 只保留中心在指定范围内的区间
            if c < c_min or c > c_max:
                continue

            # ---------- 1️⃣ 画区间 ----------
            plt.hlines(
                y=t,
                xmin=l,
                xmax=r,
                linewidth=2,
                color="#87CEFA",
                alpha=0.8
            )

            # ---------- 2️⃣ 画中点 ----------
            plt.scatter(
                c,
                t,
                s=15,
                color="#00008B",
                zorder=3  # 保证点在区间线上面
            )

    plt.xlabel("Position", fontsize=18)
    plt.ylabel("Time", fontsize=18)
    plt.title("Observation intervals and their centers", fontsize=20)

    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
