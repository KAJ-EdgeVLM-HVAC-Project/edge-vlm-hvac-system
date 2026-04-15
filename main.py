import cv2
import os
import platform
import queue
import threading
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from vlm_processor import VLMProcessor
from weather_service import WeatherService
from air_quality_service import AirQualityService
from hvac_simulator import HVACSimulator
from thermal_engine import ThermalEngine
from state_machine import StateManager, SystemState
from motion_detector import MotionDetector
from yolo_detector import YOLODetector
from pid_controller import PIDController
from sensor_interface import SensorInterface
from control_logic import decide_control, decide_window, FAN_VELOCITY
from energy_monitor import EnergyMonitor
import dashboard as dash
import user_display as udisplay

load_dotenv()

LOG_FILE      = "hvac_system_performance.csv"
SCENARIO_NAME = "Smart_Office_Initial_Test"

WEATHER_API_KEY     = os.getenv("WEATHER_API_KEY", "")
WEATHER_LAT         = 35.1044
WEATHER_LON         = 128.9750
WEATHER_FETCH_SEC   = 60

AIR_QUALITY_API_KEY = os.getenv("AIR_QUALITY_API_KEY", "")
AIR_QUALITY_STATION = os.getenv("AIR_QUALITY_STATION", "장림동")

ROOM_SIZE_M2    = 20.0
WINDOW_OPEN     = False
WORK_START_HOUR = 9
WORK_END_HOUR   = 18

YOLO_EVERY_N_FRAMES = 90  # YOLO 인원 감지 주기 (3초마다 — 30fps 기준)
PMV_UPDATE_SEC      = 5   # PMV 재계산 + PID 제어 주기 (초)


# ── CSV 초기화 / 저장 ──────────────────────────────────────────────────────────

def initialize_csv():
    if not os.path.exists(LOG_FILE):
        columns = [
            "timestamp", "scenario", "system_state",
            "out_temp", "out_humid", "out_weather", "out_wind",
            "in_temp", "in_humid",
            "people_count", "count_source", "met", "clo", "activity",
            "heat_source", "motion_score", "met_source",
            "hvac_mode", "window_rec", "room_size", "air_vel",
            "pmv_val", "comfort_status", "target_temp", "fan_speed",
            "pm10", "pm25", "khai",
        ]
        pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)


def save_log(data: dict):
    pd.DataFrame([data]).to_csv(LOG_FILE, mode="a", index=False, header=False)


# ── VLM 백그라운드 스레드 ──────────────────────────────────────────────────────

def vlm_worker(vlm, frame_lock, shared_frame_ref,
               result_queue, stop_event, interval):
    """VLM을 백그라운드에서 주기적으로 실행 — 메인 루프를 블로킹하지 않음."""
    while not stop_event.is_set():
        elapsed = 0.0
        while elapsed < interval and not stop_event.is_set():
            time.sleep(0.5)
            elapsed += 0.5
        if stop_event.is_set():
            break
        with frame_lock:
            if shared_frame_ref[0] is None:
                continue
            frame_copy = shared_frame_ref[0].copy()
        result = vlm.analyze_frame(frame_copy)
        if result is None:
            continue
        try:
            result_queue.get_nowait()   # 이전 미처리 결과 버리기
        except queue.Empty:
            pass
        result_queue.put_nowait(result)


# ── VLM 결과 처리 ─────────────────────────────────────────────────────────────

def _predict_occupancy(log_file: str) -> dict:
    """CSV 로그에서 시간대별 재실 패턴을 분석해 다음 출근 예측."""
    label_col = (100, 110, 145)
    if not os.path.exists(log_file):
        return {'message': '로그 없음 — 데이터 수집 중', 'color': label_col, 'record_count': 0}
    try:
        df = pd.read_csv(log_file)
        if 'timestamp' not in df.columns or 'people_count' not in df.columns or len(df) < 5:
            return {'message': f'기록 {len(df)}건 — 데이터 수집 중', 'color': label_col, 'record_count': len(df)}

        df['timestamp']    = pd.to_datetime(df['timestamp'], errors='coerce')
        df                 = df.dropna(subset=['timestamp'])
        df['hour']         = df['timestamp'].dt.hour
        df['people_count'] = pd.to_numeric(df['people_count'], errors='coerce').fillna(0)

        # 시간대별 재실 비율
        hourly = df.groupby('hour')['people_count'].apply(lambda x: (x > 0).mean())
        now_h  = datetime.now().hour

        # 향후 시간대 중 재실 확률 30% 이상인 첫 시간
        future = sorted(h for h in hourly.index if h > now_h and hourly[h] >= 0.3)
        if future:
            nh  = future[0]
            pct = int(hourly[nh] * 100)
            msg = f'{nh}시경 출근 예상  (재실 확률 {pct}%)'
            col = (160, 130, 255)
        elif hourly.get(now_h, 0) >= 0.3:
            pct = int(hourly[now_h] * 100)
            msg = f'현재({now_h}시) 재실 확률 {pct}% — 평균적 출근 시간대'
            col = (80, 200, 90)
        else:
            msg = f'이 시간대 재실 기록 적음 — 공실 유지 예상'
            col = label_col

        return {'message': msg, 'color': col, 'record_count': len(df)}
    except Exception as exc:
        return {'message': f'분석 오류: {str(exc)[:40]}', 'color': (210, 70, 70), 'record_count': 0}


def process_vlm_result(vlm_data, people_count, count_source,
                       motion_det, hvac, sm, engine, pid,
                       sensor, display_state,
                       out_temp, out_humid, out_weather, out_wind,
                       pm10, pm25, khai,
                       pmv_preference: float = 0.0):
    """
    VLM 분석 결과 + YOLO 인원 수를 통합하여
    PMV 계산 → 상태 머신 업데이트 → 제어 결정 → 로그 딕셔너리 반환.
    """
    # ── MET 결정: 모션 override 우선 ─────────────────────────────────────────
    if motion_det.should_override_vlm():
        effective_met = motion_det.get_motion_met()
        met_source    = "motion"
    else:
        effective_met = vlm_data["met"]
        met_source    = "vlm"

    sensor_temp, sensor_humid = sensor.read_climate()

    # 열원 감지 시 복사온도 보정
    tr_corrected = sensor_temp
    if vlm_data["heat_source"] == "yes":
        tr_corrected += VLMProcessor.TR_HEAT_OFFSET

    air_vel     = FAN_VELOCITY.get(hvac.fan_speed, 0.1)
    pmv_val     = engine.calculate_pmv(ta=sensor_temp, tr=tr_corrected,
                                       rh=sensor_humid, vel=air_vel,
                                       met=effective_met, clo=vlm_data["clo"])
    comfort_msg = engine.get_comfort_status(pmv_val)

    # ── 상태 머신 업데이트 ────────────────────────────────────────────────────
    sm.update(people_count, vlm_data["outerwear"], vlm_data["activity"])

    # ── 창문 권장 (솔루션 알림 전용, 온도 물리 미반영) ──────────────────────────
    # decide_window() 결과는 사용자에게 보여주는 권장 메시지로만 사용.
    # 실제 창문 개폐 여부는 사용자가 직접 결정 → hvac.window_open 에 적용하지 않음.
    window_rec = decide_window(pmv_val, out_temp, sensor_temp,
                               vlm_data["heat_source"], hvac.mode, people_count)

    hvac.set_room(vlm_data["room_size_m2"], False)   # window always closed for physics

    # ── 제어 결정 (사용자 선호 반영) ────────────────────────────────────────
    # adjusted_pmv: 사용자가 따뜻함 선호(+) → 시스템이 더 춥다고 인식 → 난방 강화
    adjusted_pmv = pmv_val - pmv_preference
    power, target_temp, fan_speed, mode = decide_control(
        adjusted_pmv, people_count, pid, hvac.is_on, hvac.mode,
        current_fan=hvac.fan_speed)

    if power is False:
        hvac.set_control(power=False, target=hvac.target_temp, fan=1)
        target_temp = hvac.target_temp
        fan_speed   = hvac.fan_speed
    elif power is True:
        hvac.set_control(power=True, target=target_temp, fan=fan_speed, mode=mode)
    else:
        target_temp = hvac.target_temp
        fan_speed   = hvac.fan_speed

    # ── 대시보드 표시 상태 갱신 ──────────────────────────────────────────────
    display_state.update({
        "pmv_val":       pmv_val,
        "comfort_msg":   comfort_msg,
        "people_count":  people_count,
        "count_source":  count_source,
        "activity":      vlm_data["activity"],
        "met":           effective_met,
        "clo":           vlm_data["clo"],
        "room_size":     vlm_data["room_size"],
        "room_size_m2":  vlm_data["room_size_m2"],
        "outerwear":     vlm_data["outerwear"],
        "heat_source":   vlm_data["heat_source"],
        "met_source":    met_source,
        "last_analysis": datetime.now().strftime("%H:%M:%S"),
        "pm10":          pm10,
        "pm25":          pm25,
        "khai":          khai,
    })

    # ── CSV 로그 행 반환 ──────────────────────────────────────────────────────
    return {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario":     SCENARIO_NAME,
        "system_state": sm.state.value,
        "out_temp":     out_temp,  "out_humid":  out_humid,
        "out_weather":  out_weather, "out_wind": out_wind,
        "in_temp":      sensor_temp, "in_humid": sensor_humid,
        "people_count": people_count,
        "count_source": count_source,
        "met":          effective_met, "clo": vlm_data["clo"],
        "activity":     vlm_data["activity"],
        "heat_source":  vlm_data["heat_source"],
        "motion_score": round(motion_det.current_score, 2),
        "met_source":   met_source,
        "hvac_mode":    hvac.mode,
        "window_rec":   ("open" if window_rec is True else
                         "close" if window_rec is False else "keep"),
        "room_size":    hvac.room_size,
        "air_vel":      air_vel,
        "pmv_val":      pmv_val,
        "comfort_status": comfort_msg,
        "target_temp":  target_temp,
        "fan_speed":    fan_speed,
        "pm10":         pm10,
        "pm25":         pm25,
        "khai":         khai,
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(analysis_interval: int = 30):
    initialize_csv()

    vlm         = VLMProcessor()
    weather     = WeatherService(lat=WEATHER_LAT, lon=WEATHER_LON)
    air_quality = AirQualityService(service_key=AIR_QUALITY_API_KEY,
                                    station_name=AIR_QUALITY_STATION)
    hvac        = HVACSimulator(room_size=ROOM_SIZE_M2)
    hvac.set_room(ROOM_SIZE_M2, WINDOW_OPEN)
    engine      = ThermalEngine()
    sm          = StateManager(work_start_hour=WORK_START_HOUR,
                               work_end_hour=WORK_END_HOUR)
    motion_det  = MotionDetector(history_len=10, blur_ksize=21)
    yolo        = YOLODetector(imgsz=320, conf=0.35)
    pid         = PIDController(kp=0.8, ki=0.05, kd=0.3)
    sensor      = SensorInterface(simulator=hvac)
    em          = EnergyMonitor()

    # 더미 프레임 (카메라 없을 때 공통으로 사용)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Simulation Mode", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(dummy_frame, "No Camera Available", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # macOS: CAP_AVFOUNDATION 백엔드 사용 (권한 안정성)
    # Linux/Jetson: 기본 백엔드 사용
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. 시뮬레이션 모드로 전환합니다.")
        use_camera = False
    else:
        # macOS: 첫 몇 프레임은 권한 초기화로 실패할 수 있어 워밍업
        use_camera = False
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                use_camera = True
                break
            time.sleep(0.1)
        if use_camera:
            print("카메라가 성공적으로 연결되었습니다.")
        else:
            print("카메라 프레임 읽기 실패. 카메라 권한을 확인하세요.")
            cap.release()

    # ── 스레드 설정 ───────────────────────────────────────────────────────────
    frame_lock       = threading.Lock()
    shared_frame_ref = [None]
    result_queue     = queue.Queue(maxsize=1)
    stop_event       = threading.Event()

    vlm_thread = threading.Thread(
        target=vlm_worker,
        args=(vlm, frame_lock, shared_frame_ref,
              result_queue, stop_event, analysis_interval),
        daemon=True, name="VLM-Background",
    )
    vlm_thread.start()

    # ── 대시보드 표시 상태 초기값 ─────────────────────────────────────────────
    display_state = {
        "pmv_val":      0.0,    "comfort_msg":  "분석 대기 중",
        "people_count": 0,      "count_source": "yolo",
        "activity":     "-",
        "met":          1.0,    "clo":          1.0,
        "room_size":    "medium", "room_size_m2": ROOM_SIZE_M2,
        "outerwear":    "no",   "heat_source":  "no",
        "motion_score": 0.0,    "met_source":   "vlm",
        "last_analysis": "--:--:--",
        "pm10": 0, "pm25": 0, "khai": 0,
    }

    last_people_count  = 0
    last_count_source  = "yolo"
    last_vlm_data      = None
    out_temp, out_humid, out_weather, out_wind = 20.0, 50.0, "unknown", 0.0
    pm10, pm25, khai   = 0, 0, 0
    last_weather_fetch = 0.0
    last_pmv_update    = 0.0
    frame_count        = 0

    # ── 사용자 선호 + PMV 이력 ────────────────────────────────────────────────
    pmv_preference    = 0.0          # 사용자 PMV 선호 오프셋 (-2 ~ +2)
    PMV_PREF_STEP     = 0.5          # U/D 한 번 누를 때 변화량
    PMV_PREF_MAX      = 2.0
    pmv_history: list = []           # 최근 PMV 이력 (최대 30개)
    PMV_HISTORY_MAX   = 30

    # ── 재실 예측 ─────────────────────────────────────────────────────────────
    occ_pred              = {'message': '데이터 수집 중...', 'color': (100, 110, 145), 'record_count': 0}
    occ_pred_last_update  = 0.0
    OCC_PRED_INTERVAL_SEC = 60.0

    # ── 수동 제어 상태 ────────────────────────────────────────────────────────
    manual_ctrl = {
        "enabled":     False,
        "power":       False,
        "mode":        "cool",
        "target_temp": 24.0,
        "fan_speed":   2,
    }

    # ── 환경 강제 오버라이드 (개발/테스트용) ─────────────────────────────────
    _ENV_VARS  = ["indoor_temp", "outdoor_temp", "indoor_humid", "outdoor_humid"]
    _ENV_STEPS = {"indoor_temp": 1.0, "outdoor_temp": 1.0,
                  "indoor_humid": 5.0, "outdoor_humid": 5.0}
    _ENV_LABEL = {"indoor_temp": "실내온도", "outdoor_temp": "실외온도",
                  "indoor_humid": "실내습도", "outdoor_humid": "실외습도"}
    env_override = {
        "enabled":       False,
        "indoor_temp":   22.0,
        "outdoor_temp":  20.0,
        "indoor_humid":  50.0,
        "outdoor_humid": 60.0,
        "selected":      0,
    }

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    while True:
        if use_camera:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽을 수 없습니다. 시뮬레이션으로 전환합니다.")
                use_camera = False
                frame = dummy_frame.copy()
        else:
            frame = dummy_frame.copy()

        frame_count += 1

        with frame_lock:
            shared_frame_ref[0] = frame.copy()

        motion_det.update(frame)
        display_state["motion_score"] = motion_det.current_score

        # ── YOLO 인원 감지 ────────────────────────────────────────────────────
        if frame_count % YOLO_EVERY_N_FRAMES == 0:
            if use_camera:
                yolo_count = yolo.count_people(frame)
                if yolo_count >= 0:
                    last_people_count = yolo_count
                    last_count_source = "yolo"
            else:
                # 시뮬레이션 모드: 사람 1명으로 가정 (상태 전이 테스트 가능)
                last_people_count = 1
                last_count_source = "sim"

        # ── 날씨/공기질 갱신 (60초마다) ──────────────────────────────────────
        if time.time() - last_weather_fetch >= WEATHER_FETCH_SEC:
            out_temp, out_humid, out_weather, out_wind = weather.fetch_current_weather()
            pm10, pm25, khai = air_quality.fetch_air_quality()
            last_weather_fetch = time.time()

        # ── PMV 재계산 + PID 제어 + 상태 머신 (5초마다) ──────────────────────
        now = time.time()
        if not manual_ctrl["enabled"] and now - last_pmv_update >= PMV_UPDATE_SEC:
            last_pmv_update = now

            s_temp  = (env_override["indoor_temp"]
                       if env_override["enabled"] else hvac.indoor_temp)
            s_humid = (env_override["indoor_humid"]
                       if env_override["enabled"] else hvac.indoor_humid)

            if last_vlm_data is not None:
                heat_src = last_vlm_data["heat_source"]
                eff_met  = (motion_det.get_motion_met()
                            if motion_det.should_override_vlm()
                            else last_vlm_data["met"])
                eff_clo  = last_vlm_data["clo"]
                met_src  = "motion" if motion_det.should_override_vlm() else "vlm"
            else:
                heat_src = "no"
                eff_met  = 1.2    # 기본: 기립 수준
                eff_clo  = 1.0    # 기본: 긴 소매
                met_src  = "default"

            tr_c    = s_temp + (VLMProcessor.TR_HEAT_OFFSET if heat_src == "yes" else 0.0)
            air_vel = FAN_VELOCITY.get(hvac.fan_speed, 0.1)
            pmv_now = engine.calculate_pmv(ta=s_temp, tr=tr_c,
                                           rh=s_humid, vel=air_vel,
                                           met=eff_met, clo=eff_clo)

            display_state["pmv_val"]     = pmv_now
            display_state["comfort_msg"] = engine.get_comfort_status(pmv_now)
            display_state["met"]         = eff_met
            display_state["met_source"]  = met_src

            # PMV 이력 누적
            pmv_history.append(round(pmv_now, 3))
            if len(pmv_history) > PMV_HISTORY_MAX:
                pmv_history.pop(0)

            # 에너지 모니터 틱
            em.tick(hvac.is_on, hvac.fan_speed, last_people_count, pmv_now)

            # 상태 머신 업데이트
            if last_vlm_data is not None:
                sm.update(last_people_count,
                          last_vlm_data["outerwear"],
                          last_vlm_data["activity"])
            else:
                sm.update(last_people_count)

            # 사용자 선호 반영: adjusted_pmv로 제어
            adjusted_pmv = pmv_now - pmv_preference
            power, tgt, fan, mode = decide_control(
                adjusted_pmv, last_people_count, pid, hvac.is_on, hvac.mode,
                current_fan=hvac.fan_speed)
            if power is True:
                hvac.set_control(power=True, target=tgt, fan=fan, mode=mode)
            elif power is False:
                hvac.set_control(power=False, target=hvac.target_temp, fan=1)

        # ── 수동 제어 즉시 적용 ───────────────────────────────────────────────
        if manual_ctrl["enabled"]:
            hvac.set_control(
                power  = manual_ctrl["power"],
                target = manual_ctrl["target_temp"],
                fan    = manual_ctrl["fan_speed"],
                mode   = manual_ctrl["mode"],
            )

        # ── HVAC 물리 시뮬레이션 ──────────────────────────────────────────────
        eff_out_temp  = (env_override["outdoor_temp"]
                         if env_override["enabled"] else out_temp)
        eff_out_humid = (env_override["outdoor_humid"]
                         if env_override["enabled"] else out_humid)
        hvac.simulate_step(eff_out_temp, eff_out_humid,
                           people_count=last_people_count)

        # 환경 오버라이드 시 시뮬 결과 덮어쓰기
        if env_override["enabled"]:
            hvac.indoor_temp  = env_override["indoor_temp"]
            hvac.indoor_humid = env_override["indoor_humid"]

        # ── VLM 결과 처리 ─────────────────────────────────────────────────────
        try:
            vlm_data = result_queue.get_nowait()
            last_vlm_data = vlm_data

            if not yolo.available:
                last_count_source = "vlm_fallback"

            log_row = process_vlm_result(
                vlm_data, last_people_count, last_count_source,
                motion_det, hvac, sm, engine, pid,
                sensor, display_state,
                out_temp, out_humid, out_weather, out_wind,
                pm10, pm25, khai,
                pmv_preference=pmv_preference,
            )
            save_log(log_row)
        except queue.Empty:
            pass

        # ── 재실 예측 갱신 (60초마다) ────────────────────────────────────────
        now_t = time.time()
        if now_t - occ_pred_last_update >= OCC_PRED_INTERVAL_SEC:
            occ_pred             = _predict_occupancy(LOG_FILE)
            occ_pred_last_update = now_t

        # ── 운영자 대시보드 렌더링 ────────────────────────────────────────────
        display_frame = cv2.resize(frame, (1280, 720)) if frame.shape[1] > 1280 else frame
        cam_h   = display_frame.shape[0]
        panel   = dash.build(cam_h, hvac, sm,
                             out_temp, out_humid, out_weather, out_wind,
                             display_state, manual_ctrl, env_override)
        panel_h = panel.shape[0]
        if cam_h < panel_h:
            pad        = np.zeros((panel_h - cam_h, display_frame.shape[1], 3), dtype=np.uint8)
            frame_disp = np.vstack([display_frame, pad])
        else:
            frame_disp = display_frame
        combined = np.hstack([frame_disp, panel])
        cv2.imshow("VLM HVAC — 운영자 화면", combined)

        # ── 사용자 인터페이스 창 렌더링 ───────────────────────────────────────
        em_data = {
            'saved_kwh':   round(em.get_baseline_kwh() - em.get_energy_kwh(), 4),
            'actual_kwh':  em.get_energy_kwh(),
            'savings_pct': em.get_savings_pct(),
            'comfort_pct': em.get_comfort_rate(),
        }
        user_img = udisplay.build(
            hvac, sm, display_state, em_data,
            pmv_preference, pmv_history, occ_pred, out_temp,
        )
        cv2.imshow("VLM HVAC — 사용자 인터페이스", user_img)

        # ── 키 입력 처리 ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            stop_event.set()
            vlm_thread.join(timeout=5)
            break

        elif key == ord("w"):
            hvac.window_open = not hvac.window_open
            print(f"[창문] {'열림' if hvac.window_open else '닫힘'}")

        elif key == ord("u"):
            # 사용자 선호: 더 따뜻하게
            pmv_preference = min(PMV_PREF_MAX, round(pmv_preference + PMV_PREF_STEP, 1))
            print(f"[선호] 따뜻하게 → PMV 선호 오프셋 = {pmv_preference:+.1f}")

        elif key == ord("d"):
            # 사용자 선호: 더 시원하게
            pmv_preference = max(-PMV_PREF_MAX, round(pmv_preference - PMV_PREF_STEP, 1))
            print(f"[선호] 시원하게 → PMV 선호 오프셋 = {pmv_preference:+.1f}")

        elif key == ord("s"):
            # 즉시 VLM 분석 (수동 트리거)
            with frame_lock:
                frame_copy = shared_frame_ref[0].copy()
            vlm_data = vlm.analyze_frame(frame_copy)
            if vlm_data:
                last_vlm_data = vlm_data
                log_row = process_vlm_result(
                    vlm_data, last_people_count, last_count_source,
                    motion_det, hvac, sm, engine, pid,
                    sensor, display_state,
                    out_temp, out_humid, out_weather, out_wind,
                    pm10, pm25, khai,
                    pmv_preference=pmv_preference,
                )
                save_log(log_row)

        # ── 환경 오버라이드 키 ────────────────────────────────────────────────
        elif key == ord("e"):
            env_override["enabled"] = not env_override["enabled"]
            if env_override["enabled"]:
                env_override["indoor_temp"]   = round(hvac.indoor_temp, 1)
                env_override["outdoor_temp"]  = round(out_temp, 1)
                env_override["indoor_humid"]  = round(hvac.indoor_humid, 1)
                env_override["outdoor_humid"] = round(out_humid, 1)
                print("[환경 오버라이드 ON] 현재값으로 초기화")
            else:
                print("[환경 오버라이드 OFF] 실제 시뮬레이션 복귀")

        elif env_override["enabled"] and not manual_ctrl["enabled"]:
            sel_key = _ENV_VARS[env_override["selected"]]
            step    = _ENV_STEPS[sel_key]
            if key == ord("["):
                env_override["selected"] = (env_override["selected"] - 1) % len(_ENV_VARS)
                print(f"[환경] 선택: {_ENV_LABEL[_ENV_VARS[env_override['selected']]]}")
            elif key == ord("]"):
                env_override["selected"] = (env_override["selected"] + 1) % len(_ENV_VARS)
                print(f"[환경] 선택: {_ENV_LABEL[_ENV_VARS[env_override['selected']]]}")
            elif key in (ord("="), ord("+")):
                env_override[sel_key] = round(env_override[sel_key] + step, 1)
                print(f"[환경] {_ENV_LABEL[sel_key]} = {env_override[sel_key]}")
            elif key == ord("-"):
                env_override[sel_key] = round(env_override[sel_key] - step, 1)
                print(f"[환경] {_ENV_LABEL[sel_key]} = {env_override[sel_key]}")

        # ── 수동 제어 키 ──────────────────────────────────────────────────────
        elif key == ord("m"):
            manual_ctrl["enabled"] = not manual_ctrl["enabled"]
            if manual_ctrl["enabled"]:
                manual_ctrl["power"]       = hvac.is_on
                manual_ctrl["mode"]        = hvac.mode or "cool"
                manual_ctrl["target_temp"] = hvac.target_temp
                manual_ctrl["fan_speed"]   = max(1, hvac.fan_speed)
                print("[수동 모드 ON] 현재 설정 복사 완료")
            else:
                pid.reset()
                print("[자동 모드 복귀]")

        elif manual_ctrl["enabled"]:
            if key == ord("p"):
                manual_ctrl["power"] = not manual_ctrl["power"]
                print(f"[수동] 전원 {'ON' if manual_ctrl['power'] else 'OFF'}")
            elif key == ord("c"):
                manual_ctrl["mode"]  = "cool"
                manual_ctrl["power"] = True
                print("[수동] 냉방 모드")
            elif key == ord("h"):
                manual_ctrl["mode"]  = "heat"
                manual_ctrl["power"] = True
                print("[수동] 난방 모드")
            elif key in (ord("="), ord("+")):
                manual_ctrl["target_temp"] = min(30.0, manual_ctrl["target_temp"] + 1.0)
                print(f"[수동] 설정온도 {manual_ctrl['target_temp']:.0f}°C")
            elif key == ord("-"):
                manual_ctrl["target_temp"] = max(16.0, manual_ctrl["target_temp"] - 1.0)
                print(f"[수동] 설정온도 {manual_ctrl['target_temp']:.0f}°C")
            elif key == ord("f"):
                manual_ctrl["fan_speed"] = manual_ctrl["fan_speed"] % 3 + 1
                print(f"[수동] 팬 속도 Fan {manual_ctrl['fan_speed']}")

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()
    em.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM 기반 지능형 HVAC 제어 시스템")
    parser.add_argument(
        "--interval", type=int, default=30,
        help="VLM 분석 주기(초)  기본:30 / Mac M-시리즈:10~15 / Jetson:5~10",
    )
    args = parser.parse_args()
    main(analysis_interval=args.interval)
