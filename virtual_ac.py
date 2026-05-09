from __future__ import annotations

"""
[가상 에어컨 시스템 - Virtual AC System]

카메라(VLM)로 감지한 착의량·활동량·인원·외부온도·열원 정보를 받아
현실적인 물리 모델로 실내 환경을 시뮬레이션하고 에어컨을 자동 제어합니다.

구성 클래스:
  RoomThermalModel  — 실내 열물리 시뮬레이터 (열용량·태양열·침기 포함)
  CompressorUnit    — 압축기 보호·COP·용량 변조 모델
  WindowAdvisor     — 창문 개폐 판단기 (VLM 열원·PMV·외기 복합)
  VirtualAC         — 전체 통합 제어기 (메인 루프에서 이 클래스만 사용)

사용 예시 (main.py 교체):
  ac = VirtualAC(room_size_m2=20.0)
  ac.update(
      vlm_clo=1.0, vlm_met=1.2, people_count=3,
      outdoor_temp=33.0, outdoor_humid=65.0,
      heat_source=True, pmv_val=1.4,
      weather_condition="clear"
  )
  print(ac.status_report())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ────────────────────────────────────────────────────────────
#  열거형 / 상수
# ────────────────────────────────────────────────────────────

class ACMode(Enum):
    """에어컨 운전 모드"""
    COOL     = "냉방"
    HEAT     = "난방"
    DRY      = "제습"
    FAN_ONLY = "송풍"
    AUTO     = "자동"


class ACState(Enum):
    """압축기 동작 상태"""
    OFF      = "꺼짐"
    STANDBY  = "대기"     # 전원 ON, 압축기 OFF (팬만 회전)
    STARTING = "기동중"   # 압축기 기동 대기 (최소 3분 보호)
    RUNNING  = "운전중"
    STOPPING = "정지중"   # 압축기 정지 후 최소 3분 대기


class FanSpeed(Enum):
    OFF    = 0
    LOW    = 1   # 약풍
    MID    = 2   # 중풍
    HIGH   = 3   # 강풍
    TURBO  = 4   # 터보 (도착 직후 급속 냉·난방)


# ────────────────────────────────────────────────────────────
#  실내 열물리 모델
# ────────────────────────────────────────────────────────────

@dataclass
class RoomThermalModel:
    """
    실내 열역학 시뮬레이터

    모델 가정
    ─────────
    • 실내 공기·가구·벽면을 단일 열용량 덩어리로 근사 (lumped-capacity)
    • 열용량(C) = 공기질량 × Cp + 가구·벽 보정계수
    • 열손실(UA) = 창문 + 벽 + 침기(틈새 바람)
    • 태양열 취득 = 방위각·시각·일사량 단순 모델
    • 재실자 발열 = 1인당 기초대사 80W + 활동보정

    파라미터 기본값은 사무실 20m²(천장 2.8m) 기준.
    """

    # 공간 물성
    room_size_m2:    float = 20.0   # 바닥면적 (m²)
    ceiling_h_m:     float = 2.8    # 천장 높이 (m)
    window_area_m2:  float = 3.0    # 창문 총면적 (m²)
    wall_u_value:    float = 0.35   # 외벽 열관류율 W/(m²·K)  ← 표준 단열
    window_u_closed: float = 2.8    # 창문 닫힘 열관류율 W/(m²·K)
    window_u_open:   float = 18.0   # 창문 열림 (강제 대류 근사)
    infiltration_ach: float = 0.3   # 자연 침기 환기횟수 (회/h)

    # 태양열 취득 계수 (남향 사무실 기준)
    shgc: float = 0.4      # Solar Heat Gain Coefficient
    solar_azimuth_bias: float = 0.0  # 방위각 보정 (남향=0, 동향=+90, 서향=-90)

    # 상태값 (초기값)
    indoor_temp:  float = field(default=20.0, init=False)
    indoor_humid: float = field(default=45.0, init=False)
    window_open:  bool  = field(default=False, init=False)

    def __post_init__(self):
        volume_m3 = self.room_size_m2 * self.ceiling_h_m
        air_mass_kg = volume_m3 * 1.2          # 공기 밀도 1.2 kg/m³
        air_heat_cap = air_mass_kg * 1005.0    # Cp 공기 1005 J/(kg·K)

        # 가구·벽면 열용량 추정 (공기의 약 8배 — 콘크리트·목재 가구)
        furniture_wall_cap = air_heat_cap * 8.0
        self._thermal_cap_j_per_k = air_heat_cap + furniture_wall_cap  # J/K

        # 총 열손실계수 (W/K) — 초기 계산 (창문 상태에 따라 update_step에서 재계산)
        self._wall_area_m2 = (self.room_size_m2 * 2 +
                               2 * self.ceiling_h_m * math.sqrt(self.room_size_m2) * 4) - self.window_area_m2
        self._infiltration_vol_m3 = self.room_size_m2 * self.ceiling_h_m

    # ── 외부 API ────────────────────────────────────────────

    def simulate_step(
        self,
        dt_sec: float,
        outdoor_temp: float,
        outdoor_humid: float,
        ac_heat_flux_w: float,      # AC 공급 열량 (냉방=음수, 난방=양수)
        people_count: int,
        met_avg: float,             # 평균 대사율 (met)
        heat_source_w: float = 0.0, # 추가 열원 (조리기구 등) W
        local_hour: float = 12.0,   # 현재 시각 (태양열 계산용)
    ) -> tuple[float, float]:
        """
        dt_sec 동안의 실내 온습도 변화 계산.
        Returns (indoor_temp, indoor_humid)
        """
        # 창문 상태에 따른 열손실계수
        window_u = self.window_u_open if self.window_open else self.window_u_closed
        ua_window = self.window_area_m2 * window_u
        ua_wall   = self._wall_area_m2 * self.wall_u_value
        ua_infilt  = (self._infiltration_vol_m3 * self.infiltration_ach / 3600
                      * 1.2 * 1005)  # W/K
        ua_total = ua_window + ua_wall + ua_infilt

        # 재실자 발열 (W)  — 1인당 기초 80W + 활동 보정
        # met 1.0=80W, 1.5=120W, 3.0=240W (체표면적 1.8m² 기준)
        person_heat_w = people_count * met_avg * 80.0

        # 태양열 취득 (W)
        solar_w = self._calc_solar_gain(local_hour)

        # 총 열입력 (W)
        q_total = (ac_heat_flux_w
                   + person_heat_w
                   + solar_w
                   + heat_source_w
                   - ua_total * (self.indoor_temp - outdoor_temp))

        # 온도 변화 ΔT = Q·dt / C
        delta_t = q_total * dt_sec / self._thermal_cap_j_per_k
        self.indoor_temp = round(
            max(5.0, min(50.0, self.indoor_temp + delta_t)), 2
        )

        # 습도 모델 (단순 평형)
        self._update_humidity(dt_sec, outdoor_humid, ac_heat_flux_w, people_count)

        return self.indoor_temp, self.indoor_humid

    def reset(self, init_temp: float = 20.0, init_humid: float = 45.0):
        self.indoor_temp  = init_temp
        self.indoor_humid = init_humid

    # ── 내부 메서드 ─────────────────────────────────────────

    def _calc_solar_gain(self, hour: float) -> float:
        """시각 기반 수평면 일사량 → 실내 태양열 취득 (W)"""
        # 정오 기준 가우시안 분포로 일사량 근사
        solar_hour = hour + self.solar_azimuth_bias / 15.0
        peak_irr = 800.0  # W/m² (맑은 날 최대 수평면 일사)
        irr = peak_irr * math.exp(-0.5 * ((solar_hour - 12.0) / 3.5) ** 2)
        irr = max(0.0, irr)
        return self.window_area_m2 * self.shgc * irr

    def _update_humidity(self, dt_sec: float, outdoor_humid: float,
                         ac_heat_flux_w: float, people_count: int):
        """실내 습도 업데이트 (제습·가습·침기·호흡)"""
        target_humid = self.indoor_humid

        # 창문 열림 → 외기 습도와 수렴
        if self.window_open:
            k = 0.08 * dt_sec / 60.0   # 분당 8% 수렴
            target_humid += (outdoor_humid - self.indoor_humid) * k

        # 인체 수분 발산 (1인당 약 40g/h → 대략 +0.3%RH/인/분 가정)
        person_humid_rate = people_count * 0.005 * dt_sec / 60.0
        target_humid += person_humid_rate

        # 냉방 운전 시 제습 (COP 운전 중 실내 노점 도달 → 응축수)
        if ac_heat_flux_w < -50:  # 냉방 중
            cooling_w = abs(ac_heat_flux_w)
            dehumid_rate = cooling_w * 0.0003 * dt_sec / 60.0  # 경험치
            target_humid -= dehumid_rate

        # 난방 시 상대습도 감소 (절대습도 일정, 온도↑ → RH↓)
        if ac_heat_flux_w > 50 and self.indoor_temp > 0:
            rh_correction = -0.02 * dt_sec / 60.0  # 미세 감소
            target_humid += rh_correction

        self.indoor_humid = round(
            max(15.0, min(95.0, target_humid)), 1
        )


# ────────────────────────────────────────────────────────────
#  압축기 유닛 모델
# ────────────────────────────────────────────────────────────

class CompressorUnit:
    """
    인버터 에어컨 압축기 모델

    보호 로직
    ─────────
    • 최소 ON 시간: 3분 (기동 직후 바로 끄기 방지)
    • 최소 OFF 시간: 3분 (재기동 전 냉매 압력 안정 대기)
    • 기동 지연: 5초 (STARTING → RUNNING 전환)

    COP 모델 (인버터 에어컨 근사)
    ──────────────────────────────
    COP는 실내외 온도 차이와 부하율에 따라 변함.
    냉방 COP = COP_rated × 감쇠함수(delta_T) × 부하율보정
    """

    MIN_ON_SEC    = 180   # 최소 가동 시간 (3분)
    MIN_OFF_SEC   = 180   # 최소 정지 시간 (3분)
    START_DELAY_SEC = 5   # 기동 지연

    # 정격 성능 (7kW 가정용 인버터 에어컨 기준)
    RATED_COOLING_KW = 7.0
    RATED_HEATING_KW = 8.0
    RATED_COP_COOL   = 4.2   # 냉방 정격 COP (외기 35°C 기준 CSPF)
    RATED_COP_HEAT   = 4.5   # 난방 정격 COP

    # 팬 단계별 풍량·소비전력
    FAN_AIRFLOW_M3S  = {0: 0.0,  1: 0.08, 2: 0.15, 3: 0.22, 4: 0.30}  # m³/s
    FAN_POWER_W      = {0: 0,    1: 25,   2: 55,   3: 90,   4: 130}    # W (팬 단독)

    def __init__(self):
        self.state      = ACState.OFF
        self.fan_speed  = FanSpeed.OFF
        self._load_ratio = 0.0    # 인버터 부하율 0.0~1.0

        self._state_since   = time.time()
        self._on_start_time: Optional[float] = None
        self._off_start_time: Optional[float] = time.time()

    # ── 제어 명령 ───────────────────────────────────────────

    def request_start(self) -> bool:
        """
        압축기 기동 요청.
        최소 OFF 시간을 충족하지 못하면 STANDBY 유지.
        Returns: 실제 기동 가능 여부
        """
        if self.state in (ACState.RUNNING, ACState.STARTING):
            return True
        if self.state == ACState.STOPPING:
            return False  # 아직 정지 중
        # OFF 또는 STANDBY
        if self._off_start_time and (time.time() - self._off_start_time) < self.MIN_OFF_SEC:
            self.state = ACState.STANDBY
            return False
        self._transition(ACState.STARTING)
        return True

    def request_stop(self) -> bool:
        """
        압축기 정지 요청.
        최소 ON 시간 미충족 시 무시.
        """
        if self.state == ACState.OFF:
            return True
        if self.state == ACState.STARTING:
            # 기동 중에는 바로 정지 허용 (아직 운전 미시작)
            self._transition(ACState.STANDBY)
            return True
        if self.state == ACState.RUNNING:
            if self._on_start_time and (time.time() - self._on_start_time) < self.MIN_ON_SEC:
                return False  # 최소 ON 시간 미충족
            self._transition(ACState.STOPPING)
            return True
        return True

    def power_off(self):
        """전체 전원 차단 (팬 포함)"""
        self.state = ACState.OFF
        self.fan_speed = FanSpeed.OFF
        self._load_ratio = 0.0
        self._off_start_time = time.time()

    def tick(self):
        """매 스텝 압축기 상태 업데이트 (STARTING→RUNNING, STOPPING→STANDBY 전환)"""
        elapsed = time.time() - self._state_since
        if self.state == ACState.STARTING and elapsed >= self.START_DELAY_SEC:
            self._transition(ACState.RUNNING)
        elif self.state == ACState.STOPPING and elapsed >= self.MIN_OFF_SEC:
            self._transition(ACState.STANDBY)

    # ── 성능 계산 ───────────────────────────────────────────

    def calc_heat_flux(self, mode: ACMode, load_ratio: float,
                       indoor_temp: float, outdoor_temp: float) -> float:
        """
        현재 운전 상태에서의 실내 열플럭스 계산 (W)
        Returns: 냉방 → 음수(W), 난방 → 양수(W), 미운전 → 0.0
        """
        if self.state != ACState.RUNNING:
            return 0.0

        self._load_ratio = max(0.1, min(1.0, load_ratio))

        if mode in (ACMode.COOL, ACMode.DRY):
            cop = self._cop_cooling(indoor_temp, outdoor_temp)
            cap = self.RATED_COOLING_KW * 1000 * self._load_ratio
            return -cap   # 냉방은 실내에서 열 제거 → 음수

        elif mode == ACMode.HEAT:
            cop = self._cop_heating(indoor_temp, outdoor_temp)
            cap = self.RATED_HEATING_KW * 1000 * self._load_ratio
            return cap

        else:  # FAN_ONLY
            return 0.0

    def calc_power_w(self, mode: ACMode, indoor_temp: float, outdoor_temp: float) -> float:
        """현재 소비전력 계산 (W) — 압축기 + 팬"""
        fan_w = self.FAN_POWER_W.get(self.fan_speed.value, 0)
        if self.state != ACState.RUNNING:
            return float(fan_w)

        if mode in (ACMode.COOL, ACMode.DRY):
            cop = self._cop_cooling(indoor_temp, outdoor_temp)
            cap_w = self.RATED_COOLING_KW * 1000 * self._load_ratio
        elif mode == ACMode.HEAT:
            cop = self._cop_heating(indoor_temp, outdoor_temp)
            cap_w = self.RATED_HEATING_KW * 1000 * self._load_ratio
        else:
            return float(fan_w)

        compressor_w = cap_w / max(cop, 0.5)
        return round(compressor_w + fan_w, 1)

    def get_cop(self, mode: ACMode, indoor_temp: float, outdoor_temp: float) -> float:
        if mode in (ACMode.COOL, ACMode.DRY):
            return self._cop_cooling(indoor_temp, outdoor_temp)
        elif mode == ACMode.HEAT:
            return self._cop_heating(indoor_temp, outdoor_temp)
        return 0.0

    # ── 내부 ────────────────────────────────────────────────

    def _cop_cooling(self, t_in: float, t_out: float) -> float:
        """
        냉방 COP — 실내외 온도차가 클수록 COP 저하.
        기준: 실내 27°C, 외기 35°C  → COP 4.2
        """
        delta = max(0.0, t_out - t_in)  # 보통 외기 > 실내
        # 온도차 8°C 기준, 1°C 증가마다 약 2% COP 감소
        cop = self.RATED_COP_COOL * math.exp(-0.02 * max(0.0, delta - 8.0))
        return round(max(1.0, min(6.0, cop)), 2)

    def _cop_heating(self, t_in: float, t_out: float) -> float:
        """
        난방 COP (히트펌프) — 외기가 낮을수록 COP 저하.
        기준: 실내 20°C, 외기 7°C → COP 4.5
        외기 -10°C 이하에서는 급격히 저하.
        """
        delta = max(0.0, t_in - t_out)
        cop = self.RATED_COP_HEAT * math.exp(-0.025 * max(0.0, delta - 13.0))
        # 외기 0°C 이하: 결빙 제상 로스 추가 감쇠
        if t_out < 0:
            cop *= max(0.5, 1.0 + 0.04 * t_out)
        return round(max(1.0, min(7.0, cop)), 2)

    def _transition(self, new_state: ACState):
        if new_state == self.state:
            return
        self.state = new_state
        self._state_since = time.time()
        if new_state == ACState.RUNNING:
            self._on_start_time  = time.time()
            self._off_start_time = None
        elif new_state in (ACState.STANDBY, ACState.STOPPING):
            self._off_start_time = time.time()
            self._on_start_time  = None

    @property
    def is_compressor_on(self) -> bool:
        return self.state == ACState.RUNNING


# ────────────────────────────────────────────────────────────
#  창문 개폐 판단기
# ────────────────────────────────────────────────────────────

class WindowAdvisor:
    """
    창문 개폐 추천 판단기

    판단 기준 (우선순위 순)
    ──────────────────────
    1. 열원 감지 (VLM heat_source=True) → 즉시 열기
    2. 에어컨 난방 중 → 닫기 유지
    3. 외기가 실내보다 2°C 이상 낮고 PMV > 0.5 (더운 상태) → 열기 권장
    4. 외기가 10°C 이하 → 닫기 권장
    5. 비/눈 날씨 → 닫기
    6. 실내 습도 70% 이상, 외기 습도가 낮으면 → 열기 (환기)
    """

    RAIN_KEYWORDS = {"rain", "drizzle", "thunderstorm", "snow", "sleet"}

    def __init__(self):
        self.is_open = False
        self._force_open_until: float = 0.0   # 열원 감지 시 강제 개방 타이머

    def advise(
        self,
        indoor_temp:  float,
        outdoor_temp: float,
        indoor_humid: float,
        outdoor_humid: float,
        pmv_val:       float,
        ac_mode:       ACMode,
        ac_state:      ACState,
        heat_source:   bool,
        weather_condition: str = "clear",
    ) -> bool:
        """
        창문 개폐 권장 여부 반환 (True=열기, False=닫기)
        """
        now = time.time()

        # [규칙 1] 열원 감지 → 최소 5분 강제 개방
        if heat_source:
            self._force_open_until = now + 300.0
        if now < self._force_open_until:
            return True

        # [규칙 5] 비/눈 날씨 → 무조건 닫기
        condition_lower = weather_condition.lower()
        if any(kw in condition_lower for kw in self.RAIN_KEYWORDS):
            return False

        # [규칙 4] 외기 10°C 이하 → 닫기
        if outdoor_temp <= 10.0:
            return False

        # [규칙 2] 난방 운전 중 → 닫기
        if ac_state == ACState.RUNNING and ac_mode == ACMode.HEAT:
            return False

        # [규칙 3] 더운 상태 + 외기가 훨씬 시원하면 → 열기
        if pmv_val > 0.5 and outdoor_temp < indoor_temp - 2.0 and outdoor_temp >= 15.0:
            return True

        # [규칙 6] 실내 습도 과다, 외기 습도가 낮을 때 환기
        if indoor_humid > 70.0 and outdoor_humid < indoor_humid - 15.0:
            return True

        # 그 외 → 현재 상태 유지
        return self.is_open

    def apply(self, recommendation: bool):
        self.is_open = recommendation


# ────────────────────────────────────────────────────────────
#  통합 제어기 (메인 인터페이스)
# ────────────────────────────────────────────────────────────

@dataclass
class ACStatus:
    """VirtualAC 외부 노출 상태 스냅샷"""
    mode:           str
    state:          str
    fan_speed:      int
    target_temp:    float
    indoor_temp:    float
    indoor_humid:   float
    outdoor_temp:   float
    window_open:    bool
    load_ratio:     float
    power_w:        float
    cop:            float
    energy_kwh:     float
    baseline_kwh:   float
    savings_pct:    float
    comfort_rate:   float
    window_reason:  str
    pmv_val:        float


class VirtualAC:
    """
    가상 에어컨 통합 제어기

    VLM·YOLO·날씨 데이터를 받아 에어컨 모드/팬/설정온도·창문을
    자동 결정하고 실내 환경을 시뮬레이션합니다.

    ── 자동 모드 결정 규칙 ───────────────────────────────────
    • outdoor_temp ≥ 26°C  → 냉방 우선
    • outdoor_temp ≤ 18°C  → 난방 우선
    • 18~26°C              → PMV 기반 냉·난방 자동 선택
    • PMV ∈ (-0.5, 0.5)    → 송풍(FAN_ONLY) 또는 대기

    ── 설정온도 계산 ─────────────────────────────────────────
    기본 목표 PMV=0으로 역산된 쾌적 온도를 사용하되
    VLM clo·met 값으로 개인화 보정:
      • clo 높을수록 설정온도 낮춤 (두꺼운 옷 → 더 느낌)
      • met 높을수록 설정온도 낮춤 (활동량 많으면 더 느낌)
      • 인원 많을수록 설정온도 낮춤 (체열 증가 반영)

    ── 팬 속도 결정 ──────────────────────────────────────────
    |PMV| 크기와 시스템 상태(ARRIVAL/STEADY)에 따라 1~4단 결정.
    PRE_DEPARTURE 상태면 강제 1단(절전).
    """

    # 쾌적 온도 베이스라인
    COMFORT_TEMP_COOL = 26.0   # °C (냉방 목표)
    COMFORT_TEMP_HEAT = 22.0   # °C (난방 목표)

    # 베이스라인 전력 (에너지 절약률 비교용) — 1200W 상시 가동
    BASELINE_POWER_W = 1200.0

    def __init__(
        self,
        room_size_m2: float = 20.0,
        ceiling_h_m:  float = 2.8,
        window_area_m2: float = 3.0,
    ):
        self._room   = RoomThermalModel(
            room_size_m2=room_size_m2,
            ceiling_h_m=ceiling_h_m,
            window_area_m2=window_area_m2,
        )
        self._comp   = CompressorUnit()
        self._window = WindowAdvisor()

        self._mode:        ACMode = ACMode.AUTO
        self._target_temp: float  = 24.0
        self._fan_speed:   FanSpeed = FanSpeed.OFF
        self._power_on:    bool   = False

        # 에너지 누적
        self._energy_actual_wh:   float = 0.0
        self._energy_baseline_wh: float = 0.0
        self._last_tick_time:     float = time.time()

        # 쾌적도 통계
        self._pmv_samples_total:   int = 0
        self._pmv_samples_comfort: int = 0

        # 마지막 업데이트 입력값 캐시 (status_report용)
        self._last_outdoor_temp: float = 20.0
        self._last_pmv:          float = 0.0
        self._last_power_w:      float = 0.0
        self._last_cop:          float = 0.0
        self._last_window_reason: str  = "초기화"

        # 시스템 상태 (외부에서 StateManager 연동)
        self._system_state_str: str = "STEADY"

    # ── 메인 업데이트 루프 ──────────────────────────────────

    def update(
        self,
        vlm_clo:          float,
        vlm_met:          float,
        people_count:     int,
        outdoor_temp:     float,
        outdoor_humid:    float,
        heat_source:      bool,
        pmv_val:          float,
        weather_condition: str    = "clear",
        system_state:     str     = "STEADY",
        local_hour:       float   = None,
        dt_sec:           float   = 1.0,
    ):
        """
        매 프레임 / 매 VLM 분석 후 호출.

        Args:
            vlm_clo          : VLM 감지 착의량 (0.5~1.5)
            vlm_met          : VLM 감지 대사율 (1.0~3.0)
            people_count     : YOLO 또는 VLM 감지 인원 수
            outdoor_temp     : 외부 온도 °C
            outdoor_humid    : 외부 습도 %
            heat_source      : VLM 열원 감지 여부
            pmv_val          : ThermalEngine에서 계산된 현재 PMV
            weather_condition: 날씨 상태 문자열 (e.g. "rain", "clear")
            system_state     : StateManager 상태 ("EMPTY","ARRIVAL","STEADY","PRE_DEPARTURE")
            local_hour       : 현재 시각 (None이면 time.localtime() 사용)
            dt_sec           : 시뮬레이션 스텝 간격(초)
        """
        if local_hour is None:
            local_hour = float(time.localtime().tm_hour) + time.localtime().tm_min / 60.0

        self._last_outdoor_temp  = outdoor_temp
        self._last_pmv           = pmv_val
        self._system_state_str   = system_state

        # ① 전원 / 제어 모드 결정
        self._decide_power_and_mode(system_state, outdoor_temp, pmv_val)

        # ② 설정온도 개인화 보정
        self._target_temp = self._calc_target_temp(vlm_clo, vlm_met, people_count, outdoor_temp)

        # ③ 팬 속도 결정
        self._fan_speed = self._calc_fan_speed(system_state, pmv_val)
        self._comp.fan_speed = self._fan_speed

        # ④ 압축기 기동/정지
        self._manage_compressor(system_state, pmv_val)

        # ⑤ 창문 판단
        window_rec, reason = self._advise_window(
            outdoor_temp, outdoor_humid, pmv_val,
            heat_source, weather_condition
        )
        self._window.apply(window_rec)
        self._room.window_open = window_rec
        self._last_window_reason = reason

        # ⑥ 압축기 상태 tick
        self._comp.tick()

        # ⑦ 부하율 계산
        load_ratio = self._calc_load_ratio(pmv_val)

        # ⑧ 열플럭스 계산
        heat_flux_w = self._comp.calc_heat_flux(
            self._mode, load_ratio,
            self._room.indoor_temp, outdoor_temp
        )
        heat_source_extra = 500.0 if heat_source else 0.0  # 열원 감지 시 추가 500W

        # ⑨ 실내 환경 시뮬레이션
        self._room.simulate_step(
            dt_sec       = dt_sec,
            outdoor_temp  = outdoor_temp,
            outdoor_humid = outdoor_humid,
            ac_heat_flux_w= heat_flux_w,
            people_count  = people_count,
            met_avg       = vlm_met,
            heat_source_w = heat_source_extra,
            local_hour    = local_hour,
        )

        # ⑩ 에너지·쾌적도 누적
        power_w = self._comp.calc_power_w(self._mode, self._room.indoor_temp, outdoor_temp)
        self._last_power_w = power_w
        self._last_cop     = self._comp.get_cop(self._mode, self._room.indoor_temp, outdoor_temp)

        now = time.time()
        hours = (now - self._last_tick_time) / 3600.0
        self._last_tick_time = now

        self._energy_actual_wh += power_w * hours
        if people_count > 0:
            self._energy_baseline_wh += self.BASELINE_POWER_W * hours

        self._pmv_samples_total += 1
        if -0.5 <= pmv_val <= 0.5:
            self._pmv_samples_comfort += 1

    # ── 상태 조회 ────────────────────────────────────────────

    @property
    def indoor_temp(self) -> float:
        return self._room.indoor_temp

    @property
    def indoor_humid(self) -> float:
        return self._room.indoor_humid

    @property
    def window_open(self) -> bool:
        return self._room.window_open

    @window_open.setter
    def window_open(self, val: bool):
        """main.py 'w' 키 수동 토글 호환"""
        self._room.window_open = val
        self._window.is_open   = val

    @property
    def is_on(self) -> bool:
        return self._power_on

    @property
    def fan_speed(self) -> int:
        return self._fan_speed.value

    @property
    def target_temp(self) -> float:
        return self._target_temp

    @property
    def mode(self) -> str:
        return self._mode.value

    @property
    def room_size(self) -> float:
        return self._room.room_size_m2

    def get_status(self) -> ACStatus:
        return ACStatus(
            mode          = self._mode.value,
            state         = self._comp.state.value,
            fan_speed     = self._fan_speed.value,
            target_temp   = self._target_temp,
            indoor_temp   = self._room.indoor_temp,
            indoor_humid  = self._room.indoor_humid,
            outdoor_temp  = self._last_outdoor_temp,
            window_open   = self._room.window_open,
            load_ratio    = round(self._comp._load_ratio, 2),
            power_w       = round(self._last_power_w, 1),
            cop           = self._last_cop,
            energy_kwh    = round(self._energy_actual_wh / 1000, 4),
            baseline_kwh  = round(self._energy_baseline_wh / 1000, 4),
            savings_pct   = self._get_savings_pct(),
            comfort_rate  = self._get_comfort_rate(),
            window_reason = self._last_window_reason,
            pmv_val       = self._last_pmv,
        )

    def status_report(self) -> str:
        """대시보드 출력용 멀티라인 문자열"""
        s = self.get_status()
        lines = [
            f"┌── Virtual AC Status ─────────────────────────────┐",
            f"│ 운전모드  : {s.mode:<8}  압축기: {s.state}",
            f"│ 팬속도    : {s.fan_speed}단  부하율: {s.load_ratio*100:.0f}%  COP: {s.cop:.2f}",
            f"│ 설정온도  : {s.target_temp:.1f}°C  →  실내: {s.indoor_temp:.1f}°C / {s.indoor_humid:.0f}%",
            f"│ 외기온도  : {s.outdoor_temp:.1f}°C  PMV: {s.pmv_val:+.2f}",
            f"│ 창문      : {'열림' if s.window_open else '닫힘'}  ({s.window_reason})",
            f"│ 소비전력  : {s.power_w:.0f}W  (COP {s.cop:.1f})",
            f"│ 에너지    : {s.energy_kwh:.4f}kWh  절약률: {s.savings_pct:.1f}%",
            f"│ 쾌적유지율: {s.comfort_rate:.1f}%",
            f"└──────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    # HVACSimulator 호환 API (main.py 교체 시 최소 수정)
    def set_control(self, power: bool, target: float, fan: int, mode: str = None):
        """main.py decide_control() 반환값 적용 (호환 레이어)"""
        self._power_on    = power
        self._target_temp = target
        self._fan_speed   = FanSpeed(min(4, max(0, fan)))
        self._comp.fan_speed = self._fan_speed
        if mode == "heat":
            self._mode = ACMode.HEAT
        elif mode == "cool":
            self._mode = ACMode.COOL
        if not power:
            self._comp.request_stop()
        else:
            self._comp.request_start()

    def set_room(self, size_m2: float, window_open: bool):
        self._room.room_size_m2 = size_m2
        self._room.window_open  = window_open
        self._window.is_open    = window_open

    def simulate_step(self, outdoor_temp: float, outdoor_humid: float = 50.0,
                      people_count: int = 0):
        """
        main.py 메인 루프 호환 심플 스텝 (VLM 결과 없는 매 프레임 호출용).
        세부 VLM 파라미터 없이 마지막 상태 기반으로 물리만 진행.
        """
        load_ratio = self._calc_load_ratio(self._last_pmv)
        heat_flux_w = self._comp.calc_heat_flux(
            self._mode, load_ratio,
            self._room.indoor_temp, outdoor_temp
        )
        self._comp.tick()
        self._room.simulate_step(
            dt_sec       = 0.033,   # ~30fps
            outdoor_temp  = outdoor_temp,
            outdoor_humid = outdoor_humid,
            ac_heat_flux_w= heat_flux_w,
            people_count  = people_count,
            met_avg       = 1.2,
            local_hour    = float(time.localtime().tm_hour),
        )
        power_w = self._comp.calc_power_w(self._mode, self._room.indoor_temp, outdoor_temp)
        self._last_power_w = power_w

        now = time.time()
        hours = (now - self._last_tick_time) / 3600.0
        self._last_tick_time = now
        self._energy_actual_wh += power_w * hours
        if people_count > 0:
            self._energy_baseline_wh += self.BASELINE_POWER_W * hours

        return self._room.indoor_temp, self._room.indoor_humid

    # ── 내부 결정 로직 ───────────────────────────────────────

    def _decide_power_and_mode(self, system_state: str, outdoor_temp: float, pmv_val: float):
        """시스템 상태·외기온·PMV 기반 전원/모드 결정"""
        if system_state == "EMPTY":
            self._power_on = False
            self._comp.request_stop()
            return

        self._power_on = True

        if system_state == "PRE_DEPARTURE":
            # 절전: 현재 모드 유지, 팬만 약풍
            return

        # AUTO 모드 → 냉·난방 자동 결정
        if outdoor_temp >= 26.0 or pmv_val >= 0.5:
            self._mode = ACMode.COOL
        elif outdoor_temp <= 18.0 or pmv_val <= -0.5:
            self._mode = ACMode.HEAT
        else:
            # 온화한 날씨 + 쾌적 → 송풍
            if abs(pmv_val) < 0.3:
                self._mode = ACMode.FAN_ONLY
            elif pmv_val > 0:
                self._mode = ACMode.COOL
            else:
                self._mode = ACMode.HEAT

    def _calc_target_temp(self, clo: float, met: float,
                          people_count: int, outdoor_temp: float) -> float:
        """
        VLM 착의량·활동량·인원을 반영한 개인화 설정온도.

        clo 1.0 기준 설정온도:
          냉방 26°C, 난방 22°C
        조정 규칙:
          clo 0.1 증가마다 설정온도 -0.4°C (두꺼울수록 시원하게)
          met 0.1 증가마다 설정온도 -0.3°C (활동↑ → 시원하게)
          인원 1명 추가마다 설정온도 -0.5°C (체열 증가 반영)
        """
        if self._mode in (ACMode.COOL, ACMode.DRY, ACMode.FAN_ONLY):
            base = self.COMFORT_TEMP_COOL
        else:
            base = self.COMFORT_TEMP_HEAT

        clo_adj    = -(clo - 1.0) * 4.0          # clo 1.3 → -1.2°C 보정
        met_adj    = -(met - 1.0) * 3.0          # met 1.5 → -1.5°C 보정
        occ_adj    = -max(0, people_count - 1) * 0.5
        # 외기가 매우 더울 때 설정온도 소폭 낮춤 (체감 열섬 보정)
        outdoor_adj = -max(0.0, (outdoor_temp - 32.0) * 0.3)

        adjusted = base + clo_adj + met_adj + occ_adj + outdoor_adj
        if self._mode in (ACMode.COOL, ACMode.DRY):
            adjusted = max(18.0, min(28.0, adjusted))
        else:
            adjusted = max(16.0, min(26.0, adjusted))
        return round(adjusted, 1)

    def _calc_fan_speed(self, system_state: str, pmv_val: float) -> FanSpeed:
        """PMV 절대값과 시스템 상태로 팬 속도 결정"""
        if not self._power_on:
            return FanSpeed.OFF

        if system_state == "PRE_DEPARTURE":
            return FanSpeed.LOW

        if system_state == "ARRIVAL":
            return FanSpeed.TURBO   # 도착 직후 급속 조절

        # STEADY: PMV 크기에 비례
        abs_pmv = abs(pmv_val)
        if abs_pmv < 0.3:
            return FanSpeed.LOW if self._mode != ACMode.FAN_ONLY else FanSpeed.LOW
        elif abs_pmv < 0.8:
            return FanSpeed.MID
        elif abs_pmv < 1.5:
            return FanSpeed.HIGH
        else:
            return FanSpeed.TURBO

    def _manage_compressor(self, system_state: str, pmv_val: float):
        """압축기 기동/정지 관리"""
        if not self._power_on or system_state == "EMPTY":
            self._comp.request_stop()
            return

        if self._mode == ACMode.FAN_ONLY:
            self._comp.request_stop()
            return

        # 쾌적 범위 도달 시 압축기 정지 (deadband ±0.2)
        if abs(pmv_val) < 0.2 and self._comp.is_compressor_on:
            self._comp.request_stop()
        elif abs(pmv_val) >= 0.2 and not self._comp.is_compressor_on:
            self._comp.request_start()

    def _calc_load_ratio(self, pmv_val: float) -> float:
        """PMV 오차 크기를 인버터 부하율(0.3~1.0)로 변환"""
        abs_pmv = abs(pmv_val)
        if abs_pmv < 0.2:
            return 0.0
        # 0.2~2.0 → 부하율 0.3~1.0 선형 매핑
        ratio = 0.3 + (abs_pmv - 0.2) / 1.8 * 0.7
        return round(min(1.0, ratio), 2)

    def _advise_window(
        self, outdoor_temp: float, outdoor_humid: float,
        pmv_val: float, heat_source: bool, weather_condition: str
    ) -> tuple[bool, str]:
        """창문 개폐 권장 + 이유 반환"""
        rec = self._window.advise(
            indoor_temp        = self._room.indoor_temp,
            outdoor_temp       = outdoor_temp,
            indoor_humid       = self._room.indoor_humid,
            outdoor_humid      = outdoor_humid,
            pmv_val            = pmv_val,
            ac_mode            = self._mode,
            ac_state           = self._comp.state,
            heat_source        = heat_source,
            weather_condition  = weather_condition,
        )
        # 이유 문자열
        if heat_source and time.time() < self._window._force_open_until:
            reason = "열원 감지 강제 개방"
        elif not rec and outdoor_temp <= 10.0:
            reason = "외기 저온 (닫힘)"
        elif not rec and self._comp.state == ACState.RUNNING and self._mode == ACMode.HEAT:
            reason = "난방 운전 중 (닫힘)"
        elif rec and pmv_val > 0.5:
            reason = "더운 상태 + 외기 시원 (열기)"
        elif rec and self._room.indoor_humid > 70.0:
            reason = "실내 습도 과다 (환기)"
        elif "rain" in weather_condition.lower():
            reason = "강우 감지 (닫힘)"
        else:
            reason = "유지"
        return rec, reason

    def _get_savings_pct(self) -> float:
        if self._energy_baseline_wh < 1e-6:
            return 0.0
        saved = self._energy_baseline_wh - self._energy_actual_wh
        return round(saved / self._energy_baseline_wh * 100, 1)

    def _get_comfort_rate(self) -> float:
        if self._pmv_samples_total == 0:
            return 0.0
        return round(self._pmv_samples_comfort / self._pmv_samples_total * 100, 1)
