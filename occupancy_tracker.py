"""
[객체추적 기반 재실 추정기]

YOLO 박스를 사람별 track(ID)으로 잇고, "나갔는지 / 가려졌는지"를 구분해 재실을 센다.
문 좌표를 사람이 지정할 필요가 없다 — 사람이 반복적으로 사라지는 지점을 스스로 학습해
출입구(문)로 간주한다.

── 매칭 ─────────────────────────────────────────────────────────────────────
· **헝가리안 최적 배정**(scipy) — 여러 명이 붙어 있어도 전역 최소비용으로 배정,
  그리디의 잘못된 짝짓기·연쇄 오류를 방지 (scipy 없으면 그리디로 자동 폴백)
· **이동 예측**: 직전 속도로 다음 위치를 예측해 비교 → 걷는 사람도 연결 유지
· **거리 보조**: IoU가 깨져도 중심이 가까우면 연결 (원거리 작은 박스 대응)
· **중복 병합**: 같은 사람에 두 ID가 붙으면 오래된 쪽으로 합침 (안전망)

── 출입구 자동 학습 ──────────────────────────────────────────────────────────
**재실 0↔1 전환 시점**만으로 문을 배운다. 방이 비어 있는데 사람이 나타났다면 그
자리는 반드시 출입구이고, 마지막 사람이 사라진 자리도 출입구다. (사각지대에서
다시 나타나는 경우와 섞이지 않아 오학습이 없다. 저장하지 않고 실행 때마다 학습.)
· 핫스팟 근처에서 사라짐 → **나간 것** → EXIT_CONFIRM_SEC 후 제거(−1)
· 그 외 안쪽에서 사라짐   → **가려진 것** → BLIND_HOLD_SEC 동안 유지
· 학습 전(워밍업)에는 화면 가장자리를 출구로 간주
· door_regions 인자로 문 영역을 직접 지정하면 그것을 우선 사용(옵션)

재실 = 살아있는 track 수 → 구조적으로 음수 불가.
"""

import time

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

_next_id = 0
_BIG = 1e6


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (ua + ub - inter + 1e-9)


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _size(b):
    return (max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1]))


def _anchor(box, w, h, edge_ratio=0.08):
    """출입구 학습·판정에 쓸 '사람의 위치' 기준점.

    카메라 바로 옆이 문이면 사람이 크게 잡히고 몸이 프레임 밖으로 잘린다.
    이때 박스 중심은 박스 크기에 따라 화면 중앙 쪽으로 밀려 문 위치가 망가지고,
    하체가 안 보이므로 발(하단 중앙)도 실제 접지점이 아니다.
    → 프레임 경계에 닿았다면 **닿은 경계(들)의 접촉 지점**이 곧 출입 방향이다.

      · 좌/우 + 상/하 두 경계에 동시에 닿음 → 그 **교차 코너**
      · 한 경계에만 닿음                    → 그 경계선상 박스 중앙
      · 어느 경계에도 안 닿음(문이 화면 안) → 박스 하단 중앙(발)
    """
    x1, y1, x2, y2 = box[:4]
    mx, my = w * edge_ratio, h * edge_ratio
    left, right = x1 <= mx, x2 >= w - mx
    top, bottom = y1 <= my, y2 >= h - my

    ax = x1 if left else (x2 if right else (x1 + x2) / 2.0)
    ay = y1 if top else (y2 if bottom else (y1 + y2) / 2.0)

    if (left or right) and (top or bottom):
        px, py = ax, ay                      # 교차 코너 (예: 좌측하단)
    elif left or right:
        px, py = ax, (y1 + y2) / 2.0         # 좌/우 경계선상
    elif top or bottom:
        px, py = (x1 + x2) / 2.0, ay         # 상/하 경계선상
    else:
        px, py = (x1 + x2) / 2.0, y2         # 경계 미접촉 → 발(하단 중앙)

    # 프레임 안으로 클램프 — 경계에 딱 붙으면 격자가 화면 밖으로 나가
    # 대시보드에 문이 안 보이고 판정 격자도 어긋난다.
    return (min(max(px, 1.0), w - 1.0), min(max(py, 1.0), h - 1.0))


class _ExitHotspots:
    """사람이 사라진 지점을 누적해 출입구를 스스로 학습."""

    def __init__(self, cell=80, min_hits=2, show_hits=1, decay_after=2000):
        self.cell = cell            # 격자 크기(px)
        # 판정(퇴장으로 처리)은 2회 이상 관측돼야 — 1회로 단정하면 '들어온 자리에서
        # 가려진 사람'을 나갔다고 오판한다.
        self.min_hits = min_hits
        # 표시(대시보드)는 1회부터 — 사용자가 문 인식을 바로 확인할 수 있게.
        self.show_hits = show_hits
        self.decay_after = decay_after
        self.grid = {}
        self.total = 0

    def add(self, x, y):
        k = (int(x // self.cell), int(y // self.cell))
        self.grid[k] = self.grid.get(k, 0) + 1
        self.total += 1
        if self.total % self.decay_after == 0:      # 오래된 통계 서서히 감쇠
            for kk in list(self.grid):
                self.grid[kk] *= 0.5
                if self.grid[kk] < 0.5:
                    del self.grid[kk]

    def is_exit(self, x, y):
        """해당 지점(및 인접 격자)이 학습된 출입구인가"""
        cx, cy = int(x // self.cell), int(y // self.cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if self.grid.get((cx + dx, cy + dy), 0) >= self.min_hits:
                    return True
        return False

    @property
    def learned(self):
        """표시 기준(1회 이상) — 대시보드/상태 확인용"""
        return any(v >= self.show_hits for v in self.grid.values())

    @property
    def confirmed(self):
        """판정 기준(2회 이상) — 퇴장으로 처리해도 되는 확정 출입구"""
        return any(v >= self.min_hits for v in self.grid.values())

    def confirmed_points(self):
        """판정에 쓸 '확정된' 출입구 좌표들 (min_hits 이상)"""
        c = self.cell
        return [((k[0] + .5) * c, (k[1] + .5) * c)
                for k, v in self.grid.items() if v >= self.min_hits]

    def top(self, n=3):
        items = sorted(((k, v) for k, v in self.grid.items() if v >= self.show_hits),
                       key=lambda kv: -kv[1])[:n]
        return [((k[0] + .5) * self.cell, (k[1] + .5) * self.cell, round(v, 1)) for k, v in items]


class _Track:
    def __init__(self, box, now):
        global _next_id
        _next_id += 1
        self.id = _next_id
        self.box = box
        self.vx = self.vy = 0.0
        self.first_seen = now
        self.last_seen = now
        self.confirmed = False
        self.missing_since = None
        self.exit_like = False       # 사라진 곳이 출입구였는가
        # ── 다인 분석용: 사람별 VLM 결과 캐시 ──
        self.birth_box = box               # 처음 나타난 박스(입장 지점 계산용)
        self.last_real_seen = now          # '연속 관측'의 마지막 시각(절대 상한용)
        self.missing_box = None            # 사라질 때 박스(퇴장 지점 계산용)
        self.clo = None              # 이 사람의 착의량
        self.met = None              # 이 사람의 대사율
        self.vlm_at = None           # 마지막 VLM 분석 시각(라운드로빈 선택용)

    @property
    def center(self):
        return _center(self.box)

    def predicted(self, dt):
        dx, dy = self.vx * dt, self.vy * dt
        b = self.box
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    def observe(self, box, now, was_missing=False):
        if not was_missing:
            self.last_real_seen = now      # 끊김 없이 보이는 동안만 갱신
        dt = max(1e-3, now - self.last_seen)
        ocx, ocy = self.center
        ncx, ncy = _center(box)
        self.vx = 0.5 * self.vx + 0.5 * (ncx - ocx) / dt
        self.vy = 0.5 * self.vy + 0.5 * (ncy - ocy) / dt
        self.box = box
        self.last_seen = now
        self.missing_since = None


class OccupancyTracker:

    def __init__(self,
                 count_sec: float = 0.7,
                 exit_confirm_sec: float = 15.0,
                 blind_hold_sec: float = 180.0,
                 edge_ratio: float = 0.08,
                 iou_thres: float = 0.2,
                 assoc_ratio: float = 0.8,
                 reassoc_ratio: float = 0.18,
                 merge_iou: float = 0.55,
                 max_count: int = 20,
                 max_unseen_sec: float = 600.0,   # 절대 상한(유령 track 방지)
                 door_regions=None):
        """
        door_regions: [(x1,y1,x2,y2), ...] 문 영역을 직접 지정(옵션).
                      지정하면 자동 학습보다 우선 적용된다.
        """
        self.count_sec = count_sec
        self.exit_confirm_sec = exit_confirm_sec
        self.blind_hold_sec = blind_hold_sec
        self.edge_ratio = edge_ratio
        self.iou_thres = iou_thres
        self.assoc_ratio = assoc_ratio
        self.reassoc_ratio = reassoc_ratio
        self.merge_iou = merge_iou
        self.max_count = max_count
        self.max_unseen_sec = max_unseen_sec
        self.door_regions = door_regions or []
        self.hotspots = _ExitHotspots()
        self.tracks = []
        self._visible_n = 0
        self._last_now = None
        self._seen_frames = 0
        self._prev_count  = 0
        self._fw = self._fh = 1   # 마지막 프레임 크기(기준점 계산용)

    # ── 출구 판정 ────────────────────────────────────────────────────────────
    def _is_edge(self, box, w, h):
        mx, my = w * self.edge_ratio, h * self.edge_ratio
        return (box[0] <= mx or box[1] <= my or box[2] >= w - mx or box[3] >= h - my)

    def _heading_to_door(self, t) -> bool:
        """사라질 때 '학습된 문 쪽으로 이동 중'이었는지.

        문 앞에서 검출을 놓쳐도(문에 닿기 전에 사라져도) 퇴장으로 판정하기 위함.
        가만히 있다가 사라진 사람(가려짐)과 구분된다.
        """
        speed = (t.vx ** 2 + t.vy ** 2) ** 0.5
        if speed < 20.0:                       # 거의 정지 → 가려짐으로 본다
            return False
        cx, cy = _anchor(t.box, self._fw, self._fh, self.edge_ratio)
        for dx, dy in self.hotspots.confirmed_points():
            vx_d, vy_d = dx - cx, dy - cy      # 문 방향 벡터
            dist = (vx_d ** 2 + vy_d ** 2) ** 0.5
            if dist < 1e-6:
                return True
            # 이동 방향과 문 방향의 코사인 유사도 > 0.5 (약 60도 이내)
            if (t.vx * vx_d + t.vy * vy_d) / (speed * dist) > 0.5:
                return True
        return False

    def _looks_like_exit(self, box, w, h, track=None):
        cx, cy = _anchor(box, w, h, self.edge_ratio)
        for (x1, y1, x2, y2) in self.door_regions:          # 1) 수동 지정 우선
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        if self.hotspots.is_exit(cx, cy):                    # 2) 자동 학습된 문
            return True
        if track is not None and self._heading_to_door(track):   # 3) 문 쪽으로 가던 중
            return True
        return self._is_edge(box, w, h)                      # 4) 워밍업: 가장자리

    # ── 메인 갱신 ────────────────────────────────────────────────────────────
    def update(self, boxes, frame_shape, now: float = None) -> int:
        if now is None:
            now = time.time()
        h, w = frame_shape[0], frame_shape[1]
        self._fw, self._fh = w, h
        diag = (w ** 2 + h ** 2) ** 0.5
        boxes = list(boxes or [])
        dt = 0.1 if self._last_now is None else max(1e-3, now - self._last_now)
        self._last_now = now

        nT, nD = len(self.tracks), len(boxes)

        # ── 1) 비용행렬 (작을수록 좋은 매칭) ──
        cost = [[_BIG] * nD for _ in range(nT)]
        for ti, t in enumerate(self.tracks):
            pred = t.predicted(dt) if t.missing_since is None else t.box
            tw, th = _size(t.box)
            gate = max(tw, th) * self.assoc_ratio
            if t.missing_since is not None:
                gate = max(gate, diag * self.reassoc_ratio)
            tcx, tcy = _center(pred)
            for di, b in enumerate(boxes):
                v = _iou(pred, b)
                if v >= self.iou_thres:
                    cost[ti][di] = 1.0 - v                    # IoU 매칭(비용 0~0.8)
                    continue
                dcx, dcy = _center(b)
                dist = ((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5
                bw, bh = _size(b)
                ratio = (bw * bh) / (tw * th)
                if dist <= gate and 0.25 <= ratio <= 4.0:
                    cost[ti][di] = 1.0 + dist / (gate + 1e-9)  # 거리 매칭(1~2)

        # ── 2) 헝가리안 최적 배정 (없으면 그리디 폴백) ──
        matched = {}
        if nT and nD:
            if _HAS_SCIPY:
                import numpy as np
                rows, cols = linear_sum_assignment(np.array(cost))
                for r, c in zip(rows, cols):
                    if cost[r][c] < _BIG:
                        matched[r] = c
            else:
                order = sorted(((cost[i][j], i, j) for i in range(nT) for j in range(nD)
                                if cost[i][j] < _BIG))
                usedT, usedD = set(), set()
                for _, i, j in order:
                    if i in usedT or j in usedD:
                        continue
                    matched[i] = j; usedT.add(i); usedD.add(j)

        for ti, di in matched.items():
            t = self.tracks[ti]
            t.observe(boxes[di], now, was_missing=(t.missing_since is not None))

        # ── 3) 남은 검출 → 신규 track (등장 위치도 출입구 학습에 반영) ──
        for di in set(range(nD)) - set(matched.values()):
            if len(self.tracks) < self.max_count:
                self.tracks.append(_Track(boxes[di], now))
        self._seen_frames += 1

        # ── 4) 못 잡힌 track → MISSING (사라진 지점 학습) ──
        for t in self.tracks:
            if t.last_seen == now or t.missing_since is not None:
                continue
            t.missing_since = now
            t.missing_box = t.box                # 사라질 때 박스 보관(제거 시점에 학습)
            t.exit_like = self._looks_like_exit(t.box, w, h, t)

        # ── 5) 신규 후보 승격 ──
        newly_confirmed = []
        for t in self.tracks:
            if not t.confirmed and t.missing_since is None and now - t.first_seen >= self.count_sec:
                t.confirmed = True
                newly_confirmed.append(t)

        # ── 6) 중복 병합 ──
        merged = []
        for t in sorted(self.tracks, key=lambda x: x.first_seen):
            dup = False
            for k in merged:
                if _iou(t.box, k.box) >= self.merge_iou:
                    k.confirmed = k.confirmed or t.confirmed
                    if t.missing_since is None:
                        k.missing_since = None
                    dup = True
                    break
            if not dup:
                merged.append(t)
        self.tracks = merged

        # ── 7) 정리: 출입구 소멸=퇴장(빠르게) / 그 외=가려짐(오래 유지) ──
        alive = []
        removed = []
        for t in self.tracks:
            if t.missing_since is None:
                alive.append(t); continue
            gone = now - t.missing_since
            if not t.confirmed and gone >= self.count_sec:
                continue
            # 절대 상한: 오검출 재매칭으로 유령이 살아남는 것을 원천 차단
            if now - t.last_real_seen >= self.max_unseen_sec:
                if t.confirmed: removed.append(t)
                continue
            limit = self.exit_confirm_sec if t.exit_like else self.blind_hold_sec
            if gone < limit:
                alive.append(t)
            elif t.confirmed:
                removed.append(t)
        self.tracks = alive

        # ── 8) 출입구 학습: '재실 0 ↔ 1' 전환 시점만 사용 ──
        # 방이 비어 있는데 사람이 나타났다면 그 자리는 반드시 출입구이고,
        # 마지막 사람이 사라진 자리 또한 출입구다. (사각지대에서 다시 보이는
        # 경우와 섞이지 않아 오학습이 없다.)
        cnt = self.count
        if self._prev_count == 0 and cnt >= 1:
            for t in newly_confirmed:
                self.hotspots.add(*_anchor(t.birth_box, w, h, self.edge_ratio))
        elif self._prev_count >= 1 and cnt == 0:
            for t in removed:
                if t.missing_box:
                    self.hotspots.add(*_anchor(t.missing_box, w, h, self.edge_ratio))
        self._prev_count = cnt

        self._visible_n = sum(1 for t in self.tracks
                              if t.confirmed and t.missing_since is None)
        return cnt

    @property
    def count(self) -> int:
        return sum(1 for t in self.tracks if t.confirmed)

    @property
    def is_held(self) -> bool:
        return self.count > self._visible_n

    # ── 다인 VLM 분석 지원 ──────────────────────────────────────────────────
    def pick_for_vlm(self):
        """다음에 VLM으로 분석할 사람 선택 — 보이는 track 중 가장 오래 갱신 안 된 순.

        매 주기 1명씩 라운드로빈으로 돌면 VLM 비용은 1회로 고정되면서
        (옷차림은 천천히 변하므로) 몇 주기 안에 전원이 갱신된다.
        Returns: (track_id, box) 또는 (None, None)
        """
        cands = [t for t in self.tracks if t.confirmed and t.missing_since is None]
        if not cands:
            return None, None
        cands.sort(key=lambda t: (t.vlm_at is not None, t.vlm_at or 0))
        t = cands[0]
        return t.id, t.box

    def set_person_state(self, track_id, clo, met, now=None):
        """VLM 분석 결과를 해당 사람 track에 저장."""
        for t in self.tracks:
            if t.id == track_id:
                t.clo, t.met = clo, met
                t.vlm_at = now if now is not None else time.time()
                return True
        return False

    def person_states(self):
        """분석된 사람들의 [(clo, met), ...] — 개인별 PMV 집계용."""
        return [(t.clo, t.met) for t in self.tracks
                if t.confirmed and t.clo is not None and t.met is not None]

    @property
    def door_learned(self) -> bool:
        return self.hotspots.learned

    def door_info(self):
        """학습된 출입구 후보 (x, y, 관측횟수) — 디버깅/대시보드용"""
        return self.hotspots.top()

    def door_boxes(self):
        """학습된 출입구를 화면에 그릴 사각형 [(x1,y1,x2,y2,hits), ...]"""
        out = []
        c = self.hotspots.cell
        for k, v in self.hotspots.grid.items():
            if v >= self.hotspots.show_hits:
                out.append((k[0] * c, k[1] * c, (k[0] + 1) * c, (k[1] + 1) * c, round(v, 1)))
        return out

    def reset(self):
        self.tracks = []


# ── 셀프 테스트 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    W, H = 1280, 720
    SHAPE = (H, W)
    def box(cx, cy, bw=120, bh=260):
        return (cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2, 0.9)

    print(f"=== OccupancyTracker 셀프테스트 (헝가리안={'ON' if _HAS_SCIPY else 'OFF(그리디)'}) ===")
    kw = dict(count_sec=0.5, exit_confirm_sec=3.0, blind_hold_sec=30.0)

    # 1) 빠르게 이동하는 1명 → 중복 금지
    trk = OccupancyTracker(**kw); t = 0.0; x = 300
    for _ in range(25):
        t += 0.1; x += 55; trk.update([box(x, 400)], SHAPE, now=t)
    print(f"  빠르게 이동 1명: {trk.count}명 (1)"); assert trk.count == 1

    # 2) 서로 가까이 스쳐 지나가는 2명 (헝가리안 효과)
    trk = OccupancyTracker(**kw); t = 0.0
    a, b_ = 400, 800
    for _ in range(25):
        t += 0.1; a += 30; b_ -= 30
        trk.update([box(a, 400), box(b_, 400)], SHAPE, now=t)
    print(f"  교차하는 2명: {trk.count}명 (2)"); assert trk.count == 2

    # 3) 움직이는 3명
    trk = OccupancyTracker(**kw); t = 0.0; xs = [250, 640, 1000]
    for i in range(25):
        t += 0.1; xs = [x + (30 if i % 2 == 0 else -25) for x in xs]
        trk.update([box(xs[0], 380), box(xs[1], 420), box(xs[2], 400)], SHAPE, now=t)
    print(f"  움직이는 3명: {trk.count}명 (3)"); assert trk.count == 3

    # 4) ★ 화면 '안쪽' 문(중좌측 상단)에서 반복 퇴장 → 문 자동 학습 후 빠른 퇴장 처리
    DOOR = (380, 150)
    trk = OccupancyTracker(**kw)
    t = 0.0
    for rep in range(4):                       # 4명이 같은 지점에서 사라짐
        for _ in range(8):
            t += 0.2; trk.update([box(DOOR[0], DOOR[1])], SHAPE, now=t)
        for _ in range(25):
            t += 0.2; trk.update([], SHAPE, now=t)
    print(f"  문 자동학습: {'학습됨 ✅' if trk.door_learned else '미학습'} {trk.door_info()[:1]}")
    assert trk.door_learned
    # 학습 후: 같은 지점 소멸은 '퇴장'으로 빠르게 제거
    for _ in range(8): t += 0.2; trk.update([box(DOOR[0], DOOR[1])], SHAPE, now=t)
    c1 = trk.count
    for _ in range(25): t += 0.2; trk.update([], SHAPE, now=t)   # 5초
    print(f"  학습된 문에서 소멸: {c1}명 → {trk.count}명 (0 기대)"); assert trk.count == 0

    # 5) 문이 아닌 안쪽에서 가려짐 → 유지
    trk2 = OccupancyTracker(**kw); t = 0.0
    for _ in range(8): t += 0.2; trk2.update([box(900, 500)], SHAPE, now=t)
    for _ in range(40): t += 0.2; trk2.update([], SHAPE, now=t)   # 8초
    print(f"  비출구 지점 가려짐: {trk2.count}명 (1 유지)"); assert trk2.count == 1

    # 6) 수동 문 지정
    trk3 = OccupancyTracker(door_regions=[(300, 80, 500, 250)], **kw); t = 0.0
    for _ in range(8): t += 0.2; trk3.update([box(400, 160)], SHAPE, now=t)
    for _ in range(25): t += 0.2; trk3.update([], SHAPE, now=t)
    print(f"  수동 문 지정: {trk3.count}명 (0 기대)"); assert trk3.count == 0

    print("\n✅ 모든 셀프테스트 통과")
