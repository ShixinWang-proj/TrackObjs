import numpy as np
from track import Track, TrackState
from observation import Observation
from tqdm import tqdm


class Tracker:

    def __init__(self, max_age=4, left_b=(0, 100)):
        self.tracks = []
        self.age = 0
        self.max_age = max_age
        self.records = {}
        # 左右边界
        self.left_b = left_b
        self.right_b = 2700

        # stat
        self.unmatched_ob_el = []
        self.one2many = []
        self.many2one_records = []
        self.icjc = []

    def update(self, measurements, time, sig, iou_threshold=0.1, T_vel=5):

        # ---------- 0. 初始化 ----------
        if not self.tracks:
            self._initiate(measurements)
            return

        # ---------- 1. 构造预测 & 观测 ----------
        ini_mea = [Observation(m) for m in measurements]
        energy = np.array([m.get_e(sig) for m in ini_mea])
        length = np.array([m.L for m in ini_mea])
        obs_el_labels = self._classify(np.c_[energy, length])

        obs = np.array([i.observation for i in ini_mea])
        preds = np.array([t.interval() for t in self.tracks])

        preds_el = self.get_el(preds, sig)
        preds_el_labels = self._classify(preds_el)

        matches = []
        for i, obi in enumerate(obs):
            for j, predj in enumerate(preds):
                if max(obi[0], predj[0]) <= min(obi[1], predj[1]):
                    s = self.iou(obi, predj)
                    if s >= iou_threshold:
                        matches.append((i, j, s))

        if len(matches) == 0:
            for t in self.tracks:
                t.predict()
            return

        filtered_matches_s = np.array(matches)
        filtered_matches = filtered_matches_s[:, :2].astype(int)

        i_vals = filtered_matches[:, 0]
        j_vals = filtered_matches[:, 1]

        unique_i, i_counts = np.unique(i_vals, return_counts=True)
        unique_j, j_counts = np.unique(j_vals, return_counts=True)

        i_count_dict = dict(zip(unique_i, i_counts))
        j_count_dict = dict(zip(unique_j, j_counts))

        # ==========================================================
        # 2️⃣ 处理多对多（ic>1 && jc>1） → 合并观测 → 多对一
        # ==========================================================
        group_mm = []
        used_obs = set()
        used_pred = set()

        for i, j, s in filtered_matches_s:
            i = int(i)
            j = int(j)

            if i in used_obs or j in used_pred:
                continue

            ic = i_count_dict[i]
            jc = j_count_dict[j]

            if ic > 1 and jc > 1:
                mask = (
                        (filtered_matches_s[:, 0] == i) |
                        (filtered_matches_s[:, 1] == j)
                )
                triples = filtered_matches_s[mask]

                obs_ids = sorted(set(triples[:, 0].astype(int)))
                pred_ids = sorted(set(triples[:, 1].astype(int)))

                merged_obs = (
                    min(obs[k][0] for k in obs_ids),
                    max(obs[k][1] for k in obs_ids),
                )

                group_mm.append({
                    "obs_ids": obs_ids,
                    "pred_ids": pred_ids,
                    "merged_obs": merged_obs,
                })

                used_obs.update(obs_ids)
                used_pred.update(pred_ids)

        # ==========================================================
        # 3️⃣ 剩余匹配（排除多对多）
        # ==========================================================
        mask = [
            (i not in used_obs) and (j not in used_pred)
            for i, j in filtered_matches
        ]
        filtered_matches = filtered_matches[mask]

        # ---------- unmatched ----------
        unmatched_ob = list(set(range(len(obs))) - set(filtered_matches[:, 0]) - used_obs)
        unmatched_pred = list(set(range(len(preds))) - set(filtered_matches[:, 1]) - used_pred)

        for j in unmatched_pred:
            self.tracks[j].predict()

        # ---------- 统计 ----------
        i_vals = filtered_matches[:, 0]
        j_vals = filtered_matches[:, 1]

        unique_i, i_counts = np.unique(i_vals, return_counts=True)
        unique_j, j_counts = np.unique(j_vals, return_counts=True)

        i_count_dict = dict(zip(unique_i, i_counts))
        j_count_dict = dict(zip(unique_j, j_counts))

        group_1to1 = []
        group_i_multi = []  # 多预测 → 一个观测
        group_j_multi = []  # 一个预测 → 多观测

        for i, j in filtered_matches:
            ic = i_count_dict[i]
            jc = j_count_dict[j]

            if ic == 1 and jc == 1:
                group_1to1.append((i, j))
            elif ic > 1 and jc == 1:
                group_i_multi.append((i, j))
            elif ic == 1 and jc > 1:
                group_j_multi.append((i, j))

        # ==========================================================
        # 4️⃣ 一对一
        # ==========================================================
        for i, j in group_1to1:
            self.tracks[j].update(obs[i])

        # ==========================================================
        # 5️⃣ 一个预测 → 多观测（clone）
        # ==========================================================
        groups_by_j = {}
        for i, j in group_j_multi:
            groups_by_j.setdefault(j, []).append(i)

        new_tracks = []
        for j, obs_ids in groups_by_j.items():
            if obs_el_labels[obs_ids[0]] in [2, 3]:
                self.tracks[j].update(obs[obs_ids[0]])

            for i in obs_ids[1:]:
                if obs_el_labels[i] in [2, 3]:
                    nt = self.tracks[j].clone(time)
                    nt.update(obs[i])
                    new_tracks.append(nt)

        self.tracks.extend(new_tracks)

        # ==========================================================
        # 6️⃣ 多预测 → 一个观测（交集更新）
        # ==========================================================
        groups_by_i = {}
        for i, j in group_i_multi:
            groups_by_i.setdefault(i, []).append(j)

        for i, pred_ids in groups_by_i.items():
            if obs_el_labels[i] not in [2, 3]:
                continue

            obs_i = obs[i]
            for j in pred_ids:
                pred_j = preds[j]
                left = max(obs_i[0], pred_j[0])
                right = min(obs_i[1], pred_j[1])

                if right > left:
                    self.tracks[j].update((left, right))
                else:
                    self.tracks[j].predict()

        # ==========================================================
        # 7️⃣ 多对多（已转为多对一）→ 交集更新
        # ==========================================================
        for g in group_mm:
            obs_i = g["merged_obs"]
            for j in g["pred_ids"]:
                pred_j = preds[j]
                left = max(obs_i[0], pred_j[0])
                right = min(obs_i[1], pred_j[1])

                if right > left:
                    self.tracks[j].update((left, right))
                else:
                    self.tracks[j].predict()

        # ==========================================================
        # 8️⃣ 新生轨迹
        # ==========================================================
        for i in unmatched_ob:
            if self.left_b[0] <= obs[i][1] <= self.left_b[1] and obs_el_labels[i] in [2, 3]:
                self.tracks.append(Track(obs[i]))

        # ---------- 记录历史 ----------
        alive = []
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE:
                if t.step_tentative(False) == "delete":
                    continue
            alive.append(t)

        self.tracks = alive

        for t in self.tracks:
            t.history.append(t.snapshot(time))

        self.tracks.sort(key=lambda t: t.x[0, 0])

    @staticmethod
    def _classify(el):
        """
        脉冲分三类
        :param el: ndarray, shape (N, 2)
                   第一列 e，第二列 l
        :return: list[int]，分类结果（1,2,3）
        """
        t1 = [15, 50]   # [15, 50]
        t2 = [50, 100]

        labels = []

        for e, l in el:
            if e <= t1[0] and l <= t1[1]:
                labels.append(1)
            elif e >= t2[0] and l >= t2[1]:
                labels.append(3)
            else:
                labels.append(2)

        return labels

    @staticmethod
    def get_el(pulses, sig):
        l = np.array([i[1] - i[0] for i in pulses])
        e = np.array([sum(sig[int(i[0]):int(i[1])]**2) for i in pulses])
        return np.c_[e, l]

    def predict(self, time):
        to_delete = []

        for t in self.tracks:
            if t.age <= self.max_age:
                t.predict()
            else:
                t.missed = True
                self.records[t.id] = t.history
                to_delete.append(t)

        for t in to_delete:
            self.tracks.remove(t)

    def _initiate(self, measurements):
        for m in measurements:
            self.tracks.append(Track(m))

    @staticmethod
    def iou(a, b):
        """
        a, b: 一维区间 [l, r]
        返回 IoU ∈ [0, 1]
        """
        left = max(a[0], b[0])
        right = min(a[1], b[1])

        inter = max(0.0, right - left)
        if inter == 0:
            return 0.0

        len_a = a[1] - a[0]
        len_b = b[1] - b[0]
        union = len_a + len_b - inter

        return inter / union

    @staticmethod
    def split_interval_by_ratio(I, intervals):
        """
        I: tuple (a, b)
        intervals: list of tuples [(l1, r1), (l2, r2), ..., (lk, rk)]

        return: list of k sub-intervals of I
        """
        a, b = I
        L = b - a

        lengths = [r - l for l, r in intervals]
        total = sum(lengths)

        if total <= 0:
            raise ValueError("区间总长度必须大于 0")

        result = []
        cur = a

        for li in lengths:
            seg_len = L * li / total
            next_cur = cur + seg_len
            result.append((cur, next_cur))
            cur = next_cur

        return result


if __name__ == "__main__":
    from matplotlib import rcParams
    from plot import plot_histories, plot_observations_with_centers
    rcParams['font.family'] = "Times New Roman"
    for i in tqdm(range(1, 6)):
        arr = np.load(f"D:/code/fiber/data/npy/{i}.npy")[:, 100:2800]
        Observations = np.load("D:/code/fiber/gitCode/pulseDL/Pred_intervals.npy", allow_pickle=True)[(i-1)*300:i*300]

        tracker = Tracker()
        for t, m in enumerate(Observations):
            tracker.update(m, t, arr[t])
            tracker.predict(t)

        for tr in tracker.tracks:
            tracker.records[tr.id] = tr.history

        plot_histories(tracker.records, savepath=f"tracks_v3_one2m_{i}.png")
        # plot_observations_with_centers(Observations, save_path=f"ob_with_centers_{i}.png")

