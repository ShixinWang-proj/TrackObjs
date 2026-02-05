import numpy as np


class TrackState:
    TENTATIVE = 0
    CONFIRMED = 1


class MatchMode:
    ONE2ONE = 0
    ONE2MANY = 1
    MANY2ONE = 2


class Track:
    _count = 0

    def __init__(self, interval, right_b, dt=1.0, state=TrackState.TENTATIVE, match_mode=MatchMode.ONE2ONE, confirm_threshold=7):
        self.id = Track._count
        self.right_b = right_b
        self.state = state  # 临时轨迹还是确定轨迹
        self.spawn_time = 0  # 👈 由外部赋值
        self.tentative_age = 0  # 👈 for 多尾
        Track._count += 1
        self.confirm_threshold = confirm_threshold
        self.hit_count = 0

        self.dt = dt
        self.matched = False
        self.age = 0
        self.missed = False
        self.history = []
        self.status = None
        self.match_mode = match_mode

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
            1e-4   # w
        ])
        self.R = np.diag([0.005, 0.005])

    def _predict_with_ls(self):
        """ 使用最小二乘速度预测下一帧 """
        T = self.confirm_threshold
        v_ls = self.velocity_ls(T)
        self.x[0, 0] = self.x[0, 0] + v_ls * self.dt
        self.x[1, 0] = v_ls  # 同步速度状态

    def predict(self, use_ls=False):
        if use_ls:
            self._predict_with_ls()
        else:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

    # 在 Track 类中修改 update 方法
    def update(self, interval, inflation=1.0):
        z = np.array([
            [(interval[0] + interval[1]) * 0.5],
            [interval[1] - interval[0]]
        ])

        # 核心修改：动态调整观测噪声 R
        # 如果是分裂时刻，position 突变，我们不希望轨迹被“拽”过去，
        # 所以放大 R，让滤波器“迟钝”一些，保持原有的速度方向。
        current_R = self.R.copy()
        if inflation > 1.0:
            current_R[0, 0] *= inflation  # 仅放大位置噪声，或者整体放大
            current_R[1, 1] *= inflation

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + current_R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P

        # 如果是强行分裂，为了防止下一帧搜索范围过小导致丢失，
        # 可以适当膨胀 P (增加预测的不确定性范围)
        if inflation > 1.0:
            self.P *= 1.2

        self.age = 0
        self.hit_count += 1
        self.matched = True
        if self.state == TrackState.TENTATIVE and self.hit_count >= self.confirm_threshold:
            self.state = TrackState.CONFIRMED

    def interval(self):
        c, w = self.x[0, 0], self.x[2, 0]
        return np.array([c - w / 2, c + w / 2])

    def snapshot(self, time):
        return [self.x[0, 0], self.x[2, 0], time, self.x[1, 0]]  # c, w, time, v

    def clone(self, time):
        new_track = Track(self.interval(), right_b=self.right_b, dt=self.dt,
                          state=TrackState.TENTATIVE, match_mode=MatchMode.ONE2MANY)
        new_track.x = self.x.copy()
        new_track.P = self.P.copy()
        new_track.age = 0
        new_track.tentative_age = 0
        new_track.spawn_time = time
        new_track.history = [h.copy() for h in self.history]
        return new_track

    def step_tentative(self, matched: bool, max_age=4):
        if matched:
            self.age = 0
        else:
            self.age += 1

        if self.age > max_age or self.interval()[1] > self.right_b:
            self.status = "delete"
            return

        if self.hit_count >= self.confirm_threshold:
            self.state = TrackState.CONFIRMED
            self.status = "confirmed"
            return

        self.status = "tentative"

    def mean_velocity(self, T=5):
        if len(self.history) < 2:
            return float(self.x[1, 0])
        nu = min(len(self.history), T)
        hs = self.history[-nu:]
        vs = []
        for k in range(1, len(hs)):
            c_prev, _, t_prev, _ = hs[k-1]
            c_curr, _, t_curr, _ = hs[k]
            if t_curr > t_prev:
                vs.append((c_curr - c_prev) / (t_curr - t_prev))
        return float(np.mean(vs)) if vs else float(self.x[1, 0])

    def velocity_ls(self, T=5):
        if len(self.history) < T + 1:
            return float(self.x[1, 0])
        hist = self.history[-(T + 1):]
        t = np.array([h[2] for h in hist])
        x = np.array([h[0] for h in hist])
        t_mean, x_mean = t.mean(), x.mean()
        denom = np.sum((t - t_mean) ** 2)
        if denom == 0:
            return float(self.x[1, 0])
        return float(np.sum((t - t_mean) * (x - x_mean)) / denom)

    def split_history_k(self, K):
        assert K >= 2
        histories = [[] for _ in range(K)]
        for h in self.history:
            c, w, t, v = h
            l, r = c - w / 2, c + w / 2
            L = r - l
            for m in range(K):
                lm = l + m / K * L
                rm = l + (m + 1) / K * L
                histories[m].append([0.5 * (lm + rm), rm - lm, t, v])
        return histories

    @staticmethod
    def spawn_from_history(right_b, history, base_track, time):
        c, w, _, v = history[-1]
        nt = Track((c - w / 2, c + w / 2), dt=base_track.dt,
                   state=TrackState.TENTATIVE, right_b=right_b)
        nt.x[1, 0] = v
        nt.P = base_track.P.copy()
        nt.history = [h.copy() for h in history]
        nt.spawn_time = time
        nt.tentative_age = 0
        return nt

    def get_state(self):
        c = round(float(self.x[0, 0]), 2)
        v = round(float(self.x[1, 0]), 2)
        w = round(float(self.x[2, 0]), 2)
        return {
            "id": self.id, "c": c, "v": v, "w": w,
            "interval": (c - w / 2, c + w / 2),
            "updated": self.matched, "state": self.state,
            "match_mode": self.match_mode, "age": self.age, "hit_count": self.hit_count
        }
