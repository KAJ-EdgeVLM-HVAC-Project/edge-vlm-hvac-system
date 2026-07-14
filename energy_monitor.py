"""
[에너지 모니터]
AI 제어 vs 룰베이스(고정 24°C / Fan2) 베이스라인의 소비 전력을 실시간 비교합니다.

─ AI 측  : 실제 HVACSimulator 상태(is_on, fan, target, mode, 실내온도)에서 소비 전력 적산
─ RB 측  : 내부에 별도 HVACSimulator를 두고 동일한 외기·재실 조건으로
           "재실 시 24°C 고정 + Fan2" 룰베이스 제어를 병렬 시뮬레이션
─ 쾌적율 : PMV ∈ (-0.5, +0.5) 프레임 비율 (PMV 갱신 시점마다 누적)

[전력 모델 — 압축기 부하 기반]
실제 에어컨 소비전력의 대부분은 압축기(컴프레서)가 차지하며, 송풍 팬은 수십 W
수준이다. 따라서 전력을 팬 단계만으로 매기지 않고,

    P = P_fan(fan) + P_comp_rated × 부하율(load)

으로 모델링한다. 부하율은 "실내로 새어드는 열(외기 전도 + 재실 체열)"을
"에어컨이 낼 수 있는 최대 냉난방 능력"으로 나눈 값으로, 설정온도까지 적극
구동 중이면 1.0(정격), 설정온도를 유지(데드밴드 진입)하면 열손실 상쇄분만큼의
부분부하(사이클링)가 된다. 이 구조 덕분에 설정온도·외기조건이 전력에 자연히
반영된다(낮은 설정 → 외기와 온도차↑ → 부하율↑ → 전력↑).

부하율 계산에 쓰는 상수(열시상수 τ, 냉난방률, 체열)는 HVACSimulator와
동일하게 맞춰, 시뮬레이션 물리와 전력 모델이 일관되도록 한다.
"""

from hvac_simulator import HVACSimulator

# ── 전력 모델 상수 ────────────────────────────────────────────────────────────
FAN_W        = {0: 0, 1: 40, 2: 70, 3: 110}   # 송풍 팬 전력 (실내기), W
COMP_RATED_W = 1200.0                          # 압축기 정격 입력 전력 (벽걸이 1등급급), W

# HVACSimulator와 일치시키는 물리 상수
_RATE      = 0.025      # °C/s / fan / size_factor  (COOL_RATE = HEAT_RATE)
_TAU       = 3600.0     # 건물 열시상수 (초)
_BODY_HEAT = 0.0030     # °C/s / person / size_factor (체열)
_DEADBAND  = 0.1        # 설정온도 도달 판정 폭 (°C)

RB_SETPOINT = 24.0      # 룰베이스 고정 설정 온도 (°C)
RB_FAN      = 2         # 룰베이스 재실 시 팬 속도


def hvac_watts(is_on: bool, indoor_temp: float, target_temp: float,
               fan_speed: int, mode: str, room_size: float,
               outdoor_temp: float, people_count: int) -> float:
    """공조기 순간 소비 전력 (W) — 압축기 부하 기반.

    P = 팬 전력 + 압축기 정격 × 부하율(0~1).
    AI/룰베이스 양쪽에 동일하게 적용해 공정 비교한다.
    """
    if not is_on or fan_speed <= 0:
        return 0.0

    p_fan       = FAN_W.get(fan_speed, FAN_W[3])
    size_factor = 20.0 / max(room_size, 5.0)

    # 에어컨이 낼 수 있는 최대 냉난방률 (°C/s) — 시뮬레이터와 동일
    max_drive = _RATE * fan_speed * size_factor

    # 실내 온도를 설정값에서 밀어내는 외란 (°C/s)
    conduction = (outdoor_temp - indoor_temp) / _TAU      # +면 유입, -면 유출
    body       = people_count * _BODY_HEAT * size_factor  # 항상 실내 가열(+)

    if mode == 'cool':
        # 냉방: 유입열(+conduction)과 체열이 부하로 작용
        net_gain = max(0.0, conduction) + body
        driving  = indoor_temp > target_temp + _DEADBAND
    else:
        # 난방: 유출열(-conduction)에서 체열이 상쇄
        net_gain = max(0.0, max(0.0, -conduction) - body)
        driving  = indoor_temp < target_temp - _DEADBAND

    if driving:
        load = 1.0                                        # 설정온도까지 적극 구동 = 정격
    else:
        load = min(1.0, net_gain / max_drive) if max_drive > 0 else 0.0  # 유지 = 사이클링

    return p_fan + COMP_RATED_W * load


def rb_watts(indoor_temp: float, people_count: int,
             room_size: float = 30.0, outdoor_temp: float = 24.0) -> float:
    """룰베이스(재실 시 24°C 고정 + Fan2) 순간 소비 전력 (W)."""
    if people_count <= 0:
        return 0.0
    mode = "heat" if indoor_temp < RB_SETPOINT else "cool"
    return hvac_watts(True, indoor_temp, RB_SETPOINT, RB_FAN, mode,
                      room_size, outdoor_temp, people_count)


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
        ai_w = hvac_watts(ai_hvac.is_on, ai_hvac.indoor_temp, ai_hvac.target_temp,
                          ai_hvac.fan_speed, ai_hvac.mode, ai_hvac.room_size,
                          outdoor_temp, people_count)
        rb_w = hvac_watts(self._rb.is_on, self._rb.indoor_temp, self._rb.target_temp,
                          self._rb.fan_speed, self._rb.mode, self._rb.room_size,
                          outdoor_temp, people_count)
        self.ai_wh += ai_w * dt_h
        self.rb_wh += rb_w * dt_h

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
