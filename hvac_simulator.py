import math


class HVACSimulator:
    """
    [내부 공조기 및 실내 환경 시뮬레이터]

    물리 모델:
    - 건물 열시상수 τ = 3600s (1시간) — 일반 사무실 기준
      AC 꺼진 상태에서 실내↔외기 온도차의 63%가 1시간 만에 줄어듦
    - simulate_step(dt=초)로 호출 → 카메라/영상 모드 상관없이 동일한 물리

    냉난방 속도 (Fan2, 20m² 기준):
    - 냉방: ~1.5°C/분  난방: ~3.0°C/분
    """

    TAU_BUILDING = 3600.0   # 건물 열시상수 (초), 클수록 단열 좋음
    COOL_RATE    = 0.025    # °C/s / fan_speed / size_factor (냉방)
    HEAT_RATE    = 0.025    # °C/s / fan_speed / size_factor (난방, 현실적 속도)
    BODY_HEAT    = 0.0030   # °C/s / person / size_factor   (체열)

    def __init__(self, room_size: float = 20.0):
        self.indoor_temp  = 22.0
        self.indoor_humid = 50.0

        self.room_size   = room_size
        self.window_open = False

        self.is_on       = False
        self.target_temp = 24.0
        self.fan_speed   = 1
        self.mode        = 'cool'

    def set_control(self, power: bool, target: float, fan: int, mode: str = None):
        self.is_on       = power
        self.target_temp = target
        self.fan_speed   = max(1, min(3, fan))
        if mode is not None:
            self.mode = mode

    def set_room(self, size_m2: float, window_open: bool):
        self.room_size   = size_m2
        self.window_open = window_open

    def simulate_step(self, outdoor_temp: float, outdoor_humid: float = 50.0,
                      people_count: int = 0, dt: float = 1/30):
        """
        dt: 이번 step의 실제 경과 시간 (초)
            카메라 30fps → dt=1/30
            영상 시뮬레이션 → dt=0.1 (30초 구간을 300스텝으로 분할)
        """
        size_factor = 20.0 / max(self.room_size, 5.0)

        # ── 1. 외기와의 열교환 (RC 열회로 모델) ───────────────────────────────
        # 창문 열리면 시상수 1/5로 줄어듦 (빠른 평형)
        tau = self.TAU_BUILDING / 5.0 if self.window_open else self.TAU_BUILDING
        alpha = 1.0 - math.exp(-dt / tau)
        self.indoor_temp += (outdoor_temp - self.indoor_temp) * alpha

        # ── 2. 에어컨 냉난방 ────────────────────────────────────────────────
        if self.is_on:
            if self.mode == 'heat' and self.indoor_temp < self.target_temp:
                delta = self.HEAT_RATE * self.fan_speed * size_factor * dt
                self.indoor_temp = min(self.target_temp, self.indoor_temp + delta)
                self.indoor_humid = max(25.0, self.indoor_humid - 0.03 * dt)
            elif self.mode == 'cool' and self.indoor_temp > self.target_temp:
                delta = self.COOL_RATE * self.fan_speed * size_factor * dt
                self.indoor_temp = max(self.target_temp, self.indoor_temp - delta)
                self.indoor_humid -= 0.09 * dt

        # ── 3. 창문 열릴 때 외부 습도 영향 ────────────────────────────────
        if self.window_open:
            self.indoor_humid += (outdoor_humid - self.indoor_humid) * (dt / 300.0)

        # ── 4. 재실 인원 체열 ────────────────────────────────────────────
        self.indoor_temp += people_count * self.BODY_HEAT * size_factor * dt

        self.indoor_temp  = max(10.0, min(45.0, self.indoor_temp))
        self.indoor_humid = max(15.0, min(95.0, self.indoor_humid))

        return round(self.indoor_temp, 2), round(self.indoor_humid, 1)
