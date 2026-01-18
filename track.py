import numpy as np


class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1


class Track:
    _count = 0

    def __init__(self, interval, dt=1.0, state=TrackState.CONFIRMED):
        self.id = Track._count
        self.state = state  # 👈 新增
        self.spawn_time = 0  # 👈 新增（由外部赋值）
        self.tentative_age = 0  # 👈 新增  for 多尾
        Track._count += 1

        self.dt = dt

        # -------- state: [c, v, w] --------
        c = 0.5 * (interval[0] + interval[1])
        w = interval[1] - interval[0]

        self.center = c
        self.x = np.array([
            [c],
            [15.0],
            [w]
        ])

        # -------- covariance --------
        self.P = np.diag([1.0, 1.0, 1.0])

        # -------- model matrices --------
        self.F = np.array([
            [1, dt, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        self.H = np.array([
            [1, 0, 0],
            [0, 0, 1]
        ])

        # -------- noise --------
        self.Q = np.diag([
            1e-4,  # c
            5e-4,  # v  ↑↑
            1e-4  # w
        ])
        self.R = np.diag([0.005, 0.005])

        self.age = 1
        self.missed = False
        self.history = []

    # ===============================
    # predict
    # ===============================
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    # ===============================
    # update
    # ===============================
    def update(self, interval):
        z = np.array([
            [(interval[0] + interval[1]) * 0.5],
            [interval[1] - interval[0]]
        ])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
        self.age = 0

    def interval(self):
        c, w = self.x[0, 0], self.x[2, 0]
        return np.array([c - w / 2, c + w / 2])

    def snapshot(self, time):
        return [self.x[0, 0], self.x[2, 0], time, self.x[1, 0]]  # c,w

    def clone(self, time):
        new_track = Track(self.interval(), dt=self.dt,
                          state=TrackState.TENTATIVE)

        new_track.x = self.x.copy()
        new_track.P = self.P.copy()

        new_track.age = 0
        new_track.tentative_age = 0
        new_track.spawn_time = time

        new_track.history = [h.copy() for h in self.history]

        return new_track

    def step_tentative(self, is_many_to_one, T_window=5):
        """
        is_many_to_one: bool
            当前时刻是否仍然处于 多对一匹配
        """
        if self.state != TrackState.TENTATIVE:
            return "confirmed"

        self.tentative_age += 1

        # 只要出现过 非多对一
        if not is_many_to_one:
            self.state = TrackState.CONFIRMED
            return "confirmed"

        # 时间窗口耗尽，且全是多对一
        if self.tentative_age >= T_window:
            return "delete"

        return "tentative"

    def mean_velocity(self, T=5):
        """
        返回时间窗口内的平均速度
        """
        if len(self.history) < 2:
            return float(self.x[1, 0])

        # 取最近 nu 帧
        nu = min(len(self.history), T)
        hs = self.history[-nu:]

        # 用中心差分估计速度
        vs = []
        for k in range(1, len(hs)):
            c_prev = hs[k - 1][0]
            c_curr = hs[k][0]
            t_prev = hs[k - 1][2]
            t_curr = hs[k][2]
            if t_curr > t_prev:
                vs.append((c_curr - c_prev) / (t_curr - t_prev))

        if len(vs) == 0:
            return float(self.x[1, 0])

        return float(np.mean(vs))

    def velocity_ls(self, T=5):
        """
        最小二乘速度估计
        使用最近 T+1 个位置点
        """
        if len(self.history) < T + 1:
            return float(self.x[1, 0])

        # 最近 T+1 帧
        hist = self.history[-(T + 1):]

        # 时间（用相对时间即可）
        t = np.array([h[2] for h in hist])

        # 位置：中心 c
        x = np.array([h[0] for h in hist])

        t_mean = t.mean()
        x_mean = x.mean()

        denom = np.sum((t - t_mean) ** 2)
        if denom == 0:
            return float(self.x[1, 0])

        v_T = np.sum((t - t_mean) * (x - x_mean)) / denom
        return float(v_T)

    def split_history_k(self, K):
        """
        将 history 按区间等比例分裂成 K 条
        return: List[K][history]
        """
        assert K >= 2

        # 初始化 K 条 history
        histories = [[] for _ in range(K)]

        for h in self.history:
            c, w, t, v = h
            l = c - w / 2
            r = c + w / 2
            L = r - l

            for m in range(K):
                lm = l + m / K * L
                rm = l + (m + 1) / K * L
                cm = 0.5 * (lm + rm)
                wm = rm - lm

                histories[m].append([cm, wm, t, v])

        return histories

    @staticmethod
    def spawn_from_history(history, base_track, time):
        """
        从一条分裂后的 history 生成 Track
        """
        c, w, _, v = history[-1]
        interval = (c - w / 2, c + w / 2)

        nt = Track(interval,
                   dt=base_track.dt,
                   state=TrackState.TENTATIVE)

        # 继承速度
        nt.x[1, 0] = v

        # 协方差放大（不确定性）
        nt.P = base_track.P.copy()
        nt.P[0, 0] *= 1.5
        nt.P[2, 2] *= 1.5

        nt.history = [h.copy() for h in history]
        nt.spawn_time = time
        nt.tentative_age = 0

        return nt
