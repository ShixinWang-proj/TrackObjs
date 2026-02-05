import numpy as np
from tqdm import tqdm
from track import Track, TrackState, MatchMode


class Tracker:
    def __init__(self, max_age=3, left_b=(0, 300)):
        self.tracks = []
        self.max_age = max_age
        self.records = {}
        self.left_b = left_b
        self.right_b = 2700

        # 统计项
        self.unmatched_ob_el = []
        self.one2many = []
        self.many2one_records = []
        self.icjc = []

    def update(self, measurements, time, sig, iou_threshold=0.1, merge_threshold=13):
        for tr in self.tracks:
            tr.matched = False

        # ---------- 0️⃣ 过滤越界观测 ----------
        measurements = [m for m in measurements if 50 < m[1] < self.right_b-50]

        if len(measurements) == 0: return

        # ---------- 0. 初始化 ----------
        if not self.tracks:
            self._initiate(measurements)
            return

        # ---------- 1. 构造预测 & 观测分类 ----------
        # 此处简化 Observation 处理逻辑
        obs = np.array([m for m in measurements])
        energy_list = [np.sum(sig[int(m[0]):int(m[1])] ** 2) for m in measurements]
        length_list = [m[1] - m[0] for m in measurements]
        obs_el_labels = self._classify(np.c_[energy_list, length_list])

        preds = np.array([t.interval() for t in self.tracks])

        matches = []
        for i, obi in enumerate(obs):
            for j, predj in enumerate(preds):
                if max(obi[0], predj[0]) <= min(obi[1], predj[1]):
                    s = self.iou(obi, predj)
                    if s >= iou_threshold:
                        matches.append((i, j, s))

        if len(matches) == 0: return

        filtered_matches_s = np.array(matches)
        filtered_matches = filtered_matches_s[:, :2].astype(int)

        i_count_dict = dict(zip(*np.unique(filtered_matches[:, 0], return_counts=True)))
        j_count_dict = dict(zip(*np.unique(filtered_matches[:, 1], return_counts=True)))

        # ---------- 2️⃣ 处理多对多 -> 转为多对一 ----------
        group_mm, used_obs, used_pred = [], set(), set()
        for i, j, s in filtered_matches_s:
            i, j = int(i), int(j)
            if i in used_obs or j in used_pred: continue
            if i_count_dict[i] > 1 and j_count_dict[j] > 1:
                mask = (filtered_matches_s[:, 0] == i) | (filtered_matches_s[:, 1] == j)
                triples = filtered_matches_s[mask]
                o_ids = sorted(set(triples[:, 0].astype(int)))
                p_ids = sorted(set(triples[:, 1].astype(int)))
                merged_obs = (min(obs[k][0] for k in o_ids), max(obs[k][1] for k in o_ids))
                group_mm.append({"obs_ids": o_ids, "pred_ids": p_ids, "merged_obs": merged_obs})
                used_obs.update(o_ids)
                used_pred.update(p_ids)

        # ---------- 3️⃣ 剩余匹配分类 ----------
        mask = [(i not in used_obs and j not in used_pred) for i, j in filtered_matches]
        filtered_matches = filtered_matches[mask]
        unmatched_ob = list(set(range(len(obs))) - set(filtered_matches[:, 0]) - used_obs)

        group_1to1, group_i_multi, group_j_multi = [], [], []
        i_counts = dict(zip(*np.unique(filtered_matches[:, 0], return_counts=True)))
        j_counts = dict(zip(*np.unique(filtered_matches[:, 1], return_counts=True)))

        for i, j in filtered_matches:
            ic, jc = i_counts[i], j_counts[j]
            if ic == 1 and jc == 1:
                group_1to1.append((i, j))
            elif ic > 1 and jc == 1:
                group_i_multi.append((i, j))
            elif ic == 1 and jc > 1:
                group_j_multi.append((i, j))

        # ---------- 4️⃣ 一对一更新 ----------
        for i, j in group_1to1:
            self.tracks[j].update(obs[i])
            self.tracks[j].match_mode = MatchMode.ONE2ONE

            # ==========================================================
            # 5️⃣ 一对多（分支验证 + 惯性保护）
            # ==========================================================
            groups_by_j = {}
            for i, j in group_j_multi:
                groups_by_j.setdefault(j, []).append(i)

            new_tracks = []

            for j, obs_ids in groups_by_j.items():
                tr = self.tracks[j]

                # 1. 筛选有效观测
                valid_obs_ids = [i for i in obs_ids if obs_el_labels[i] in [2, 3]]
                if not valid_obs_ids:
                    # 如果没有有效长脉冲，就按普通最近邻更新一个即可，或者跳过
                    continue

                # 2. 排序：找中心距离最近的，作为主轨迹的继承者
                def get_distance(obs_idx):
                    obs_c = 0.5 * (obs[obs_idx][0] + obs[obs_idx][1])
                    pred_c = tr.x[0, 0]
                    return abs(obs_c - pred_c)

                valid_obs_ids.sort(key=get_distance)

                # 3. 主轨迹更新 (Inherit)
                # ✨ 关键点：分裂时，观测中心必然偏离原合并中心。
                # 使用 inflation=10.0 (甚至更大)，告诉 KF：
                # "虽然我把这个观测给了你，但位置可能不准，请保持你原有的速度方向！"
                best_idx = valid_obs_ids[0]
                tr.update(obs[best_idx], inflation=10.0)
                tr.match_mode = MatchMode.ONE2MANY

                # 4. 分支克隆 (Clone)
                for o_id in valid_obs_ids[1:]:
                    # Clone 时，历史轨迹是完全复制的
                    branch_track = tr.clone(time)

                    # A. 状态设为 TENTATIVE，重置 hit_count
                    branch_track.state = TrackState.TENTATIVE
                    branch_track.hit_count = 1

                    # B. 分支更新
                    # 对于分支，因为它代表分离出去的新物体，它的位置确实是新的观测位置。
                    # 但因为它继承了母体的速度（可能很大），为了防止它第一帧就乱飞，
                    # 给它一个适度的 inflation (如 5.0)，让它在位置上稍微“软”着陆，
                    # 或者完全信任观测 (inflation=1.0)，取决于你对新分支的信任度。
                    # 建议：也稍微 inflate 一下，防止速度突变过大
                    branch_track.update(obs[o_id], inflation=5.0)

                    branch_track.match_mode = MatchMode.ONE2MANY
                    new_tracks.append(branch_track)

            self.tracks.extend(new_tracks)

        # ---------- 6️⃣ 多对一 & 7️⃣ 多对多（交集更新） ----------
        groups_by_i = {}
        for i, j in group_i_multi: groups_by_i.setdefault(i, []).append(j)
        for i, p_ids in groups_by_i.items():
            if obs_el_labels[i] not in [2, 3]: continue
            for j in p_ids:
                l, r = max(obs[i][0], preds[j][0]), min(obs[i][1], preds[j][1])
                self.tracks[j].match_mode = MatchMode.MANY2ONE
                if r > l: self.tracks[j].update((l, r))

        for g in group_mm:
            for j in g["pred_ids"]:
                l, r = max(g["merged_obs"][0], preds[j][0]), min(g["merged_obs"][1], preds[j][1])
                self.tracks[j].match_mode = MatchMode.MANY2ONE
                if r > l: self.tracks[j].update((l, r))

        # ---------- 8️⃣ 新生轨迹 ----------
        for i in unmatched_ob:
            if self.left_b[0] <= obs[i][1] <= self.left_b[1] and obs_el_labels[i] in [2, 3]:
                self.tracks.append(Track(obs[i], right_b=self.right_b))

        # ---------- 记录历史与清理 ----------
        alive = []
        for tr in self.tracks:
            tr.step_tentative(tr.matched, max_age=self.max_age)
            if tr.status == "delete":
                self.records[tr.id] = tr.history
                continue
            alive.append(tr)
            tr.history.append(tr.snapshot(time))
        self.tracks = alive
        self.tracks.sort(key=lambda t: t.x[0, 0])

    def predict(self):
        for tr in self.tracks:
            if tr.match_mode == MatchMode.MANY2ONE and not tr.matched: continue
            use_ls = (tr.status == "confirmed") and (not tr.matched)
            tr.predict(use_ls)

    @staticmethod
    def _classify(el):
        t1, t2 = [10, 35], [50, 100]
        labels = []
        for e, l in el:
            if e <= t1[0] and l <= t1[1]:
                labels.append(1)
            elif e >= t2[0] and l >= t2[1]:
                labels.append(3)
            else:
                labels.append(2)
        return labels

    def _initiate(self, measurements):
        for m in measurements:
            self.tracks.append(Track(m, right_b=self.right_b))

    @staticmethod
    def iou(a, b):
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        if inter == 0: return 0.0
        return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter)

    def get_track_state_by_id(self, tid):
        for t in self.tracks:
            if t.id == tid: return t.get_state()
        return None

