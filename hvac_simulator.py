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

    # ── 습도 모델 ───────────────────────────────────────────────────────────
    # 냉방은 코일 응축으로 제습하지만 무한정 건조해지지 않는다. 침기·재실자
    # 수분 발생과 균형을 이뤄 45~55%에서 평형을 이룬다(공조 실내 통상값).
    # 단방향 감소로 두면 장시간 운전 시 하한에 붙고, 과건조 상태는 증발
    # 열손실을 키워 PMV를 실제보다 시원하게 만든다.
    COOL_EQUIL_RH = 50.0    # 냉방 평형 상대습도 (%)
    HEAT_EQUIL_RH = 40.0    # 난방 평형 상대습도 (%) — 가열로 더 건조
    HUMID_TAU     = 900.0   # 공조 제습/가습 시상수 (초)
    INFILT_TAU    = 7200.0  # 창문 닫힘 상태의 침기 시상수 (초)
    BODY_MOIST    = 0.0008  # %RH/s / person / size_factor  (재실자 수분 발생)

    def __init__(self, room_size: float = 20.0):
        self.indoor_temp  = 22.0
        self.indoor_humid = 50.0

        self.room_size   = room_size
        self.window_open = False

        self.is_on       = False
        self.fan_only    = False   # 압축기 정지 + 송풍만 가동
        self.target_temp = 24.0
        self.fan_speed   = 1
        self.mode        = 'cool'

    def set_control(self, power: bool, target: float, fan: int, mode: str = None,
                    fan_only: bool = False):
        """fan_only=True 면 압축기는 멈추고 송풍만 유지한다.

        실제 에어컨은 설정온도 도달 후에도 송풍을 계속 돌린다(압축기만 사이클링).
        송풍을 함께 끄면 기류가 사라져 체감이 급격히 나빠지므로, 재실 중에는
        송풍을 유지하는 편이 쾌적·에너지 양쪽에서 유리하다.
        """
        self.fan_only    = bool(fan_only) and not power
        self.is_on       = power
        self.target_temp = target
        self.fan_speed   = max(1, min(3, fan))
        if mode is not None:
            self.mode = mode

    def set_room(self, size_m2: float, window_open: bool):
        self.room_size   = size_m2
        self.window_open = window_open

    def _relax_humid(self, target_rh: float, dt: float):
        """공조 가동 중 실내 습도를 평형값으로 지수 수렴시킨다."""
        self.indoor_humid += (target_rh - self.indoor_humid) * (
            1.0 - math.exp(-dt / self.HUMID_TAU))

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
        # 습도는 '평형값으로 수렴'시킨다. 과거에는 냉방 중 매 스텝 일정량을 빼기만
        # 해서(단방향 감소) 장시간 운전 시 하한 15%에 붙어버렸고, 건조할수록 증발
        # 열손실이 커져 PMV가 실제보다 시원하게 계산되는 문제가 있었다.
        # 실제 냉방 실내는 코일 응축(제습)과 침기·재실자 수분 발생이 균형을 이뤄
        # 45~55% 부근에서 평형을 이룬다.
        if self.is_on:
            if self.mode == 'heat' and self.indoor_temp < self.target_temp:
                delta = self.HEAT_RATE * self.fan_speed * size_factor * dt
                self.indoor_temp = min(self.target_temp, self.indoor_temp + delta)
                self._relax_humid(self.HEAT_EQUIL_RH, dt)
            elif self.mode == 'cool' and self.indoor_temp > self.target_temp:
                delta = self.COOL_RATE * self.fan_speed * size_factor * dt
                self.indoor_temp = max(self.target_temp, self.indoor_temp - delta)
                self._relax_humid(self.COOL_EQUIL_RH, dt)

        # ── 3. 외부 습도 영향 (창문 개방 시 급속, 닫혀 있어도 침기로 완만) ──
        tau_h = 300.0 if self.window_open else self.INFILT_TAU
        self.indoor_humid += (outdoor_humid - self.indoor_humid) * (
            1.0 - math.exp(-dt / tau_h))

        # 송풍 전용: 냉난방은 없고 기류만 유지 (온도 변화 없음)

        # ── 4. 재실 인원 체열 + 수분 발생 ────────────────────────────────
        self.indoor_temp  += people_count * self.BODY_HEAT * size_factor * dt
        self.indoor_humid += people_count * self.BODY_MOIST * size_factor * dt

        self.indoor_temp  = max(10.0, min(45.0, self.indoor_temp))
        self.indoor_humid = max(20.0, min(95.0, self.indoor_humid))

        return round(self.indoor_temp, 2), round(self.indoor_humid, 1)
