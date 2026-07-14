"""
[검출-앵커 재실 추정기]

YOLO 순간 오검출/미검출로 인원이 튀는 것을 막고, "잠깐 잡힌 것"과 "실제 재실"을
구분한다. 재실 인원을 어떤 값 d로 바꾸려면 **d가 일정 시간 연속 유지**되어야 한다.

── 변경 규칙 (재실 → 새 검출값 d) ──────────────────────────────────────────────
· 상승 (d > 재실)      : d가 count_sec(기본 1초) 이상 지속돼야 반영
                          → "잠깐 잡힌 것"은 카운트하지 않음
· 하강 (0 < d < 재실)  : d가 hold_sec(기본 60초) 이상 유지돼야 반영
                          → 예: 2명이 3명 됐다가 다시 2명으로 1분 유지되면 2명으로
· 완전 공실 (d = 0)    : empty_confirm_sec(기본 120초) 무검출 + 무모션이라야 0
                          → 화면에 안 보여도 움직임 있으면 계속 재실 유지
· 문 라인 입/퇴장       : (옵션) 유예 없이 즉시 ±1

d가 현재 재실과 같아지면 후보를 초기화(안정)한다. 값이 오락가락하면 반영 안 됨.
"""

import time


class OccupancyTracker:

    def __init__(self,
                 count_sec: float = 1.0,
                 hold_sec: float = 60.0,
                 empty_confirm_sec: float = 120.0,
                 motion_threshold: float = 1.2,
                 max_count: int = 20):
        """
        Args:
            count_sec         : 인원 증가를 인정하기까지 필요한 연속 감지 시간(초)
            hold_sec          : 인원 감소(>0)를 반영하기까지 필요한 유지 시간(초)
            empty_confirm_sec : 완전 공실(0) 확정까지 무검출·무모션 지속(초)
            motion_threshold  : 이 이상이면 '움직임 있음'(MotionDetector.current_score 기준)
            max_count         : 재실 상한
        """
        self.count_sec         = count_sec
        self.hold_sec          = hold_sec
        self.empty_confirm_sec = empty_confirm_sec
        self.motion_threshold  = motion_threshold
        self.max_count         = max_count

        self.occupancy      = 0
        self._cand          = None    # 변경 후보값
        self._cand_since    = 0.0     # 후보가 관측되기 시작한 시각
        self._last_detected = 0

    def update(self, detected_count, motion_score: float = 0.0,
               now: float = None, entered: int = 0, exited: int = 0) -> int:
        if now is None:
            now = time.time()
        motion = motion_score >= self.motion_threshold

        # ── 문 라인 이벤트(옵션): 즉시 반영 ──
        if entered:
            self.occupancy = min(self.max_count, self.occupancy + int(entered))
            self._cand = None
        if exited:
            self.occupancy = max(0, self.occupancy - int(exited))
            self._cand = None

        # ── 검출 정보 없음 → 유지 ──
        if detected_count is None or detected_count < 0:
            return self.occupancy

        d = min(int(detected_count), self.max_count)
        self._last_detected = d

        # ── 현재와 같으면 안정 (후보 초기화) ──
        if d == self.occupancy:
            self._cand = None
            return self.occupancy

        # ── 변경 후보 타이머 ──
        if self._cand != d:
            self._cand       = d
            self._cand_since = now

        if d > self.occupancy:
            required = self.count_sec              # 상승: 1초 지속
        elif d == 0:
            required = self.empty_confirm_sec      # 완전 공실
        else:
            required = self.hold_sec               # 하강(>0): 60초 유지

        # 공실(0)로 갈 때 움직임 있으면 보류 (누가 움직이는데 놓친 것)
        if d == 0 and motion:
            self._cand_since = now
            return self.occupancy

        if now - self._cand_since >= required:
            self.occupancy = d
            self._cand = None
        return self.occupancy

    def reset(self, count: int = 0):
        """재실 인원 강제 설정(수동 리셋)."""
        self.occupancy = max(0, min(int(count), self.max_count))
        self._cand = None

    @property
    def count(self) -> int:
        return self.occupancy

    @property
    def is_held(self) -> bool:
        """검출보다 재실이 높음 = 안 보이는 사람 유지 중 (대시보드 표시용)."""
        return self.occupancy > self._last_detected


# ── 셀프 테스트 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def run(name, events, seed=0, **kw):
        trk = OccupancyTracker(**kw)
        if seed:
            trk.reset(seed)
        t = 0.0; traj = []
        for dt, d, m in events:
            t += dt
            traj.append(trk.update(d, m, now=t))
        print(f"{name:34s} → {traj}")
        return traj

    print("=== OccupancyTracker 셀프테스트 ===")
    kw = dict(count_sec=1.0, hold_sec=5.0, empty_confirm_sec=8.0)

    # 1) 0.3초짜리 순간 감지 반복 → 1초 미만이라 카운트 안 됨
    t1 = run("잠깐 잡힘(0.3s)×깜빡 → 0 유지",
             [(0.3,1,0),(0.3,0,0)]*6, **kw)
    assert max(t1) == 0, "잠깐 감지가 카운트됨"

    # 2) 1초 이상 지속 → 카운트
    t2 = run("1명 1.5초 지속 → 1",
             [(0.3,1,0)]*6, **kw)
    assert t2[-1] == 1, "지속 감지 카운트 실패"

    # 3) 2명→3명(1초 지속)→다시 2명 1분 유지 → 2
    ev = [(0.3,3,0)]*5 + [(0.3,2,0)]*20     # 3을 1.5초, 그다음 2를 6초
    t3 = run("2→3→2 유지 → 3 올랐다 2로", ev, seed=2, **kw)
    assert 3 in t3 and t3[-1] == 2, f"하향 보정 실패: {t3[-1]}"

    # 4) 공실: 1명→0 지속(무모션) → empty_confirm 후 0
    t4 = run("1명→0 지속(무모션) → 0",
             [(0.5,1,0)]*3 + [(0.5,0,0)]*20, **kw)
    assert t4[-1] == 0, "공실 확정 실패"

    # 5) 0 검출이지만 모션 있음 → 유지
    t5 = run("미검출+모션 → 유지",
             [(0.5,1,0)]*3 + [(0.5,0,5)]*20, **kw)
    assert t5[-1] >= 1, "모션 유지 실패"

    print("\n✅ 모든 셀프테스트 통과")
