"""
핵심 도메인 로직 단위 테스트 (하드웨어·torch·cv2 의존 없음).

CI(GitHub Actions, Ubuntu)에서 pytest만으로 실행 가능하도록
순수 파이썬 모듈(thermal_engine / control_logic / pid_controller /
hvac_simulator)만 검증한다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thermal_engine import ThermalEngine
from pid_controller import PIDController
from control_logic import decide_control, COMFORT_TEMP
from hvac_simulator import HVACSimulator


# ── ThermalEngine (PMV / ISO 7730) ──────────────────────────────────────────

def test_pmv_within_bounds():
    """PMV는 항상 [-3, 3] 범위로 클램핑된다."""
    eng = ThermalEngine()
    for ta in (10, 24, 40):
        pmv = eng.calculate_pmv(ta=ta, tr=ta, rh=50, vel=0.1, met=1.2, clo=1.0)
        assert -3.0 <= pmv <= 3.0


def test_pmv_hot_is_positive():
    """더운 조건(고온·두꺼운 옷)은 양(+)의 PMV."""
    eng = ThermalEngine()
    pmv = eng.calculate_pmv(ta=32, tr=32, rh=60, vel=0.1, met=1.2, clo=1.0)
    assert pmv > 0.5


def test_pmv_cold_is_negative():
    """추운 조건(저온·얇은 옷)은 음(-)의 PMV."""
    eng = ThermalEngine()
    pmv = eng.calculate_pmv(ta=16, tr=16, rh=40, vel=0.1, met=1.0, clo=0.5)
    assert pmv < -0.5


def test_comfort_status_neutral():
    """중립 PMV(0)는 '쾌적' 상태로 분류된다."""
    eng = ThermalEngine()
    assert "쾌적" in eng.get_comfort_status(0.0)


# ── PIDController ────────────────────────────────────────────────────────────

def test_pid_target_is_neutral():
    assert PIDController.PMV_TARGET == 0.0


def test_pid_output_to_fan_speed():
    assert PIDController.output_to_fan_speed(0.4) == 1    # round(0.4)=0 → 하한 1
    assert PIDController.output_to_fan_speed(3.0) == 3    # 상한 3
    assert PIDController.output_to_fan_speed(1.6) == 2


def test_pid_deadband_returns_zero():
    """deadband 이내의 미세 오차는 출력 0."""
    pid = PIDController()
    out = pid.compute(0.0, dt=1.0)   # 오차 0 → deadband 내
    assert out == 0.0


def test_pid_reset():
    pid = PIDController()
    pid.compute(2.0, dt=1.0)
    pid.reset()
    assert pid.integral == 0.0


# ── control_logic.decide_control ────────────────────────────────────────────

def test_empty_room_turns_off():
    """공실(인원 0)이면 전원 OFF."""
    pid = PIDController()
    power, target, fan, mode = decide_control(0.5, people_count=0, pid=pid, dt=5.0)
    assert power is False
    assert target == COMFORT_TEMP


def test_hot_occupied_turns_on_cooling():
    """재실 + 매우 더움 → 냉방 ON."""
    pid = PIDController()
    power, target, fan, mode = decide_control(
        2.0, people_count=2, pid=pid, hvac_mode="cool", dt=5.0, indoor_temp=28.0)
    assert power is True
    assert mode == "cool"
    assert 1 <= fan <= 3


def test_cold_occupied_turns_on_heating():
    """재실 + 매우 추움 → 난방 ON."""
    pid = PIDController()
    power, target, fan, mode = decide_control(
        -2.0, people_count=1, pid=pid, hvac_mode="heat", dt=5.0, indoor_temp=18.0)
    assert power is True
    assert mode == "heat"


# ── HVACSimulator ────────────────────────────────────────────────────────────

def test_simulator_cools_when_ac_on():
    """냉방 가동 시 실내온도가 외기·목표 방향으로 내려간다."""
    hvac = HVACSimulator(room_size=20.0)
    hvac.indoor_temp = 30.0
    hvac.set_control(power=True, target=22.0, fan=3, mode="cool")
    for _ in range(200):
        hvac.simulate_step(28.0, 50.0, people_count=1, dt=1.0)
    assert hvac.indoor_temp < 30.0
