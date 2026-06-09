from __future__ import annotations

"""
[제어 로직 모듈]
decide_control / decide_window 함수를 main.py 와 scenario_runner.py 가
공유할 수 있도록 분리합니다.

─ main.py         : dt=None → time.time() 기반 실시간 PID
─ scenario_runner : dt=5.0  → 시뮬레이션 고정 주기(5 sim-초)

──────────────────────────────────────────────────────────────────
[핵심 설계]

목표온도 동적 설정 (Dynamic Target Temperature)
  고정 24°C 대신 PMV 심각도에 비례해 목표 온도를 낮추거나 높임.

  이유: hvac_simulator는 indoor_temp > target_temp 일 때만 냉방,
        indoor_temp < target_temp 일 때만 난방을 수행.
        target=24°C 고정이면, 실내가 24°C 에 도달한 순간 AC가 멈추지만
        clo·met이 높으면 PMV가 여전히 +0.8 이상인 경우가 발생.
        → target을 낮게 설정해 물리 냉방이 계속되도록 허용.

  과냉 방지: 히스테리시스(PMV_OFF=0.2)가 실제 종료 조건.
        실내가 target에 도달하기 전에 PMV가 쾌적 범위에 들어오면
        AC가 먼저 꺼지므로 실제로 18°C까지 내려가는 일은 없음.

팬 속도 하한 보장 (PMV-proportional Fan Minimum)
  PID 출력만으로는 PMV가 매우 클 때도 Fan1~2에 머무를 수 있음.
  (kp=0.8 기준: PMV=3.0 → PID output=-2.4 → Fan2)
  PMV 절대값에 따라 팬 최솟값을 직접 보장해 응답성 확보.
──────────────────────────────────────────────────────────────────
"""

from pid_controller import PIDController

# FAN_VELOCITY: fan_speed(1~3) → 기류 속도(m/s) 매핑
FAN_VELOCITY: dict[int, float] = {1: 0.1, 2: 0.3, 3: 0.5}

# 열원(조리기구 등) 감지 시 복사온도 보정값 (VLMProcessor.TR_HEAT_OFFSET 과 동일)
TR_HEAT_OFFSET: float = 4.0

# 쾌적 기준 온도 (PMV 쾌적 구간 유지 목표)
COMFORT_TEMP: float = 24.0

# 재실 시 난방 최소 유지 온도 — 겨울 두꺼운 옷(clo=1.0)이면 PMV가 19°C에서도
# 쾌적(-0.5~+0.5)으로 나와 AI가 조기 OFF됨. 실제 사무실 기준(22°C)을 하한으로 보장.
HEAT_FLOOR_TEMP: float = 22.0

# ── 목표 온도 테이블 ──────────────────────────────────────────────────────────dsaf
# (PMV 임계값, 목표 온도) — 리스트는 높은 PMV 순으로 정렬
# PMV가 임계값 이상이면 해당 목표온도로 적극 냉/난방
_COOL_TARGETS: list[tuple[float, float]] = [
    (2.0, 18.0),   # 매우 더움  → 18°C 까지 강냉
    (1.0, 20.0),   # 더움       → 20°C 까지 냉방
    (0.5, 22.0),   # 조금 더움  → 22°C 까지 냉방
]
_HEAT_TARGETS: list[tuple[float, float]] = [
    (-2.0, 30.0),  # 매우 추움  → 30°C 까지 강난방
    (-1.0, 28.0),  # 추움       → 28°C 까지 난방
    (-0.5, 26.0),  # 조금 추움  → 26°C 까지 난방
]

# ── 히스테리시스 임계값 ───────────────────────────────────────────────────────
PMV_ON:  float = 0.5   # AC 켜는 PMV 절대값
PMV_OFF: float = 0.2   # AC 끄는 PMV 절대값 (PMV_ON > PMV_OFF → oscillation 방지)


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _target_temp(pmv_val: float, mode: str) -> float:
    """PMV 심각도와 모드에 따른 목표온도 결정."""
    if mode == "cool":
        for threshold, target in _COOL_TARGETS:
            if pmv_val >= threshold:
                return target
    else:
        for threshold, target in _HEAT_TARGETS:
            if pmv_val <= threshold:
                return target
    return COMFORT_TEMP


def _min_fan_from_pmv(pmv_val: float) -> int:
    """
    PMV 절대값 기반 팬 속도 하한.

    Fan1은 혹서(35°C 외기, 6인)·혹한(-3°C 외기) 조건에서 균형점이
    실내온도 출발점과 거의 같아 온도를 실질적으로 바꾸지 못한다.
    (Fan1 여름 균형점 ≈ 32.3°C / 겨울 균형점 ≈ 10.3°C)

    → PMV ≥ 0.5 (AC 가동 기준 그 자체)에서도 Fan2 이상을 보장해
       어떤 날씨 조건에서도 냉·난방이 실제로 온도를 변화시키도록 한다.
    """
    a = abs(pmv_val)
    if a >= 1.5:   # 강한 냉/난방 필요 → Fan3
        return 3
    if a >= 0.5:   # AC 가동 기준(PMV_ON=0.5)과 동일: Fan2 이상
        return 2
    return 1


# ── 공개 API ──────────────────────────────────────────────────────────────────

def decide_control(pmv_val: float, people_count: int, pid: PIDController,
                   hvac_is_on: bool = False, hvac_mode: str = "cool",
                   dt: float = None, current_fan: int = 1,
                   indoor_temp: float = None):
    """
    PMV 기반 제어 결정 (동적 목표온도 + PMV 비례 팬 속도 + 히스테리시스).

    Args:
        pmv_val     : 현재 PMV 지수 (-3.0 ~ +3.0)
        people_count: 재실 인원 (0 이면 즉시 OFF)
        pid         : PIDController 인스턴스 (팬 속도 계산용)
        hvac_is_on  : 현재 AC 켜짐 여부 (히스테리시스 판단용)
        hvac_mode   : 현재 AC 모드 ('cool'|'heat')
        dt          : PID 경과 시간(초). None → time.time() 자동. 시나리오 러너는 5.0.
        current_fan : 현재 HVAC 팬 속도(1~3). 히스테리시스 구간에서 그대로 유지.
                      ── PMV ↔ 팬속도 ↔ 기류속도 피드백 진동 방지 ──
                      팬 속도가 PMV 계산에 영향(공기 속도) → PMV 변화 → 팬 속도 재결정
                      이 루프가 히스테리시스 구간에서 Fan1↔Fan2를 5초 간격으로 진동시켜
                      결과적으로 평균 냉방/난방 출력이 반감되는 현상을 막는다.

    Returns:
        (power: bool, target_temp: float, fan_speed: int, mode: str|None)
    """
    # 공실 → 즉시 OFF
    if people_count == 0:
        pid.reset()
        return False, COMFORT_TEMP, 1, None

    # 난방 최소온도 보장 — 겨울 두꺼운 옷 착용 시 PMV가 19~20°C에서도 쾌적으로
    # 나와 AI가 조기 OFF됨. 실내온도 < HEAT_FLOOR_TEMP이면 24°C까지 계속 난방.
    if indoor_temp is not None and indoor_temp < HEAT_FLOOR_TEMP:
        fan_floor = max(_min_fan_from_pmv(pmv_val), 2)
        return True, COMFORT_TEMP, fan_floor, "heat"

    # 팬 속도: PID 출력과 PMV 비례 하한 중 큰 값
    pid_output = pid.compute(pmv_val, dt=dt)
    fan = max(
        PIDController.output_to_fan_speed(pid_output),
        _min_fan_from_pmv(pmv_val),
        1,
    )

    # ── 냉방 ──────────────────────────────────────────────────────────────────
    if pmv_val > PMV_ON:
        return True, _target_temp(pmv_val, "cool"), fan, "cool"

    if hvac_is_on and hvac_mode == "cool" and pmv_val > PMV_OFF:
        # 히스테리시스 유지 — current_fan 그대로 유지(진동 방지)
        # 목표온도를 24°C 로 올려 시뮬레이터가 추가 냉방 없이 온도를 자연 상승시키도록 허용
        return True, COMFORT_TEMP, current_fan, "cool"

    # ── 난방 ──────────────────────────────────────────────────────────────────
    if pmv_val < -PMV_ON:
        return True, _target_temp(pmv_val, "heat"), fan, "heat"

    if hvac_is_on and hvac_mode == "heat" and pmv_val < -PMV_OFF:
        # 히스테리시스 유지 — current_fan 그대로 유지(진동 방지)
        return True, COMFORT_TEMP, current_fan, "heat"

    # ── 쾌적 구간 → OFF ───────────────────────────────────────────────────────
    pid.reset()
    return False, COMFORT_TEMP, 1, None


def decide_window(pmv_val: float, outdoor_temp: float,
                  indoor_temp: float, heat_source: str,
                  hvac_mode: str, people_count: int) -> bool | None:
    """
    창문 개폐 판단.

    우선순위:
      1. 공실           → 닫기
      2. 열원 감지       → 열기 (환기 최우선)
      3. 난방 중         → 닫기 (열 손실 방지)
      4. 매우 더움 + 실외 시원 → 열기 (자연 환기로 냉방 보조)
      5. 조금 더움 + 실외 충분히 시원 → 열기
      6. 그 외           → 현재 상태 유지

    Returns:
        True  : 열기
        False : 닫기
        None  : 현재 상태 유지
    """
    # 공실
    if people_count == 0:
        return False

    # 열원(조리기구 등) → 환기 최우선
    if heat_source == "yes":
        return True

    # 난방 중 → 창문 닫기 (열 손실 방지)
    if hvac_mode == "heat":
        return False

    # 매우 더움 (PMV > 1.0): 실외가 조금만 시원해도 적극 환기
    if pmv_val > 1.0 and outdoor_temp < indoor_temp - 1.5:
        return True

    # 조금 더움 (PMV > 0.5): 실외가 충분히 시원할 때 환기
    if pmv_val > 0.5 and outdoor_temp < indoor_temp - 3.0:
        return True

    return None
