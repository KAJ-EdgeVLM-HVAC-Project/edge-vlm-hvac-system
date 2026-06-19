"""
[에너지 모니터]
AI 제어 vs 룰베이스(고정 24°C / Fan2) 베이스라인의 소비 전력을 실시간 비교합니다.

─ AI 측  : 실제 HVACSimulator 상태(is_on, fan_speed)에서 소비 전력 적산
─ RB 측  : 내부에 별도 HVACSimulator를 두고 동일한 외기·재실 조건으로
           "재실 시 24°C 고정 + Fan2" 룰베이스 제어를 병렬 시뮬레이션
─ 쾌적율 : PMV ∈ (-0.5, +0.5) 프레임 비율 (PMV 갱신 시점마다 누적)

video_mode(main.py)의 RB 베이스라인 모델과 동일한 상수를 사용:
  - POWER_W: Fan 1/2/3 = 800/1200/1600 W
  - 서모스탯 사이클링: 설정온도 ±0.5°C 도달 시 25% 소비(팬+대기)
"""

from hvac_simulator import HVACSimulator

# fan_speed → 소비 전력 (W). 0 = OFF
POWER_W = {0: 0, 1: 800, 2: 1200, 3: 1600}

RB_SETPOINT = 24.0   # 룰베이스 고정 설정 온도 (°C)
RB_FAN      = 2      # 룰베이스 재실 시 팬 속도


def rb_watts(indoor_temp: float, people_count: int) -> float:
    """룰베이스 순간 소비 전력 (서모스탯 사이클링 반영)."""
    if people_count <= 0:
        return 0.0
    at_setpoint = abs(indoor_temp - RB_SETPOINT) <= 0.5
    return POWER_W[RB_FAN] * (0.25 if at_setpoint else 1.0)


class EnergyMonitor:
    def __init__(self, room_size: float = 20.0,
                 init_temp: float = None, init_humid: float = None):
        self._rb = HVACSimulator(room_size=room_size)
        if init_temp is not None:
            self._rb.indoor_temp = init_temp
        if init_humid is not None:
            self._rb.indoor_humid = init_humid

        self.ai_wh = 0.0
        self.rb_wh = 0.0
        self._comfort_frames = 0
        self._total_frames   = 0

    # ── 주기 호출 API ─────────────────────────────────────────────────────────

    def sync_room(self, size_m2: float):
        """VLM이 감지한 방 크기를 RB 시뮬레이터에도 동일 적용."""
        self._rb.set_room(size_m2, False)

    def step(self, dt_sec: float, outdoor_temp: float, outdoor_humid: float,
             people_count: int, ai_hvac: HVACSimulator):
        """매 프레임 호출 — AI/RB 양쪽 에너지 적산 + RB 물리 1스텝.

        Args:
            dt_sec       : 이번 스텝 경과 시간 (초)
            ai_hvac      : 실제(AI 제어) HVACSimulator — 전력 계산에만 읽음
        """
        # RB 제어: 재실 시 24°C 고정 + Fan2, 공실 시 OFF
        if people_count > 0:
            mode = "heat" if self._rb.indoor_temp < RB_SETPOINT else "cool"
            self._rb.set_control(power=True, target=RB_SETPOINT,
                                 fan=RB_FAN, mode=mode)
        else:
            self._rb.set_control(power=False, target=RB_SETPOINT, fan=1)

        self._rb.simulate_step(outdoor_temp, outdoor_humid,
                               people_count=people_count, dt=dt_sec)

        dt_h = dt_sec / 3600.0
        ai_w = POWER_W.get(ai_hvac.fan_speed, 0) if ai_hvac.is_on else 0
        self.ai_wh += ai_w * dt_h
        self.rb_wh += rb_watts(self._rb.indoor_temp, people_count) * dt_h

    def update_comfort(self, pmv: float, occupied: bool = True):
        """PMV 갱신 시점마다 호출 — 재실 중 쾌적율 누적."""
        if not occupied:
            return
        self._total_frames += 1
        if -0.5 < pmv < 0.5:
            self._comfort_frames += 1

    # ── 조회 ──────────────────────────────────────────────────────────────────

    @property
    def saved_wh(self) -> float:
        return self.rb_wh - self.ai_wh

    @property
    def savings_pct(self) -> float:
        if self.rb_wh <= 0:
            return 0.0
        return (self.rb_wh - self.ai_wh) / self.rb_wh * 100.0

    @property
    def comfort_rate(self) -> float:
        """재실 중 PMV 쾌적 구간 비율 (0~100%)."""
        if self._total_frames == 0:
            return 0.0
        return self._comfort_frames / self._total_frames * 100.0

    @property
    def rb_indoor_temp(self) -> float:
        return self._rb.indoor_temp
