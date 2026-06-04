import cv2
import os
try:
    import platform as _plat
    from pynput import keyboard as _kb
    _PYNPUT_OK = _plat.system() != "Darwin"  # 맥에서는 HIToolbox 스레드 충돌로 비활성화
except Exception:
    _kb = None
    _PYNPUT_OK = False
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
from env_profiles import PROFILES, EnvProfile
from startup_screen import show_and_select, StartupResult
import dashboard as dash
import user_display as udisplay

load_dotenv()


def _build_vlm_window(vlm_data, vlm_time, analyzing: bool) -> "np.ndarray":
    """VLM 분석 결과 표시 창 (480x320)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import platform, os

    W, H = 520, 340
    img  = Image.new("RGB", (W, H), (15, 15, 25))
    draw = ImageDraw.Draw(img)

    # 폰트
    def _font(sz):
        sys = platform.system()
        cands = []
        if sys == "Linux":
            cands = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        elif sys == "Darwin":
            cands = ["/System/Library/Fonts/AppleSDGothicNeo.ttc"]
        for p in cands:
            try: return ImageFont.truetype(p, sz)
            except: pass
        return ImageFont.load_default()

    f14 = _font(14); f12 = _font(12); f11 = _font(11)

    # 헤더
    draw.rectangle([(0,0),(W,36)], fill=(30,30,60))
    draw.text((12, 9), "VLM Context", font=f14, fill=(180,200,255))
    status_col = (80,220,120) if not analyzing else (255,200,60)
    status_txt = "분석중..." if analyzing else ("완료" if vlm_data else "대기중")
    draw.text((W-90, 11), status_txt, font=f12, fill=status_col)

    y = 46
    if vlm_time:
        draw.text((12, y), f"마지막 분석: {vlm_time}", font=f11, fill=(140,140,160))
        y += 20

    if vlm_data:
        raw = vlm_data.get("raw_response", "")
        # raw JSON 표시 (최대 3줄)
        draw.text((12, y), "[ Raw Output ]", font=f12, fill=(100,180,255))
        y += 18
        max_w = 58
        for i, line in enumerate(raw.split("\n")[:4]):
            if len(line) > max_w: line = line[:max_w] + "..."
            draw.text((14, y), line, font=f11, fill=(200,200,180))
            y += 15

        y += 6
        draw.line([(12, y), (W-12, y)], fill=(50,50,70))
        y += 8

        # 파싱 결과
        draw.text((12, y), "[ Parsed ]", font=f12, fill=(100,180,255))
        y += 18
        fields = [
            ("activity",    vlm_data.get("activity", "-")),
            ("clo",         f"{vlm_data.get('clo', 0):.2f}"),
            ("met",         f"{vlm_data.get('met', 0):.1f}"),
            ("room_size",   vlm_data.get("room_size", "-")),
            ("heat_source", vlm_data.get("heat_source", "-")),
            ("outerwear",   vlm_data.get("outerwear", "-")),
        ]
        for k, v in fields:
            draw.text((16, y), f"{k:<12}: {v}", font=f11, fill=(220,220,200))
            y += 15
            if y > H - 10:
                break

    return np.array(img)[:,:,::-1]  # RGB→BGR



def _is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release")


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

def _seasonal_clo(profile: EnvProfile) -> float:
    """계절(월)에 따른 CLO 기본값 반환 — VLM 실패 시 fallback 전용"""
    m = datetime.now().month
    if 6 <= m <= 8:
        return profile.clo_summer
    if m in (3, 4, 5, 9, 10, 11):
        return profile.clo_spring_fall
    return profile.clo_winter


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
    # ── 시작 화면 (모드 + 환경 선택) ──────────────────────────────────────────
    startup = show_and_select()

    # 영상 모드로 선택됐으면 video_mode로 넘김
    if startup.mode == 'video' and startup.video_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(startup.video_path))[0]
        out_dir = os.path.join("results", f"{base}_{ts}")
        video_mode(startup.video_path, analysis_interval, out_dir)
        return

    env_profile = startup.profile

    initialize_csv()

    # ── 로딩 화면 ─────────────────────────────────────────────────────────────
    _load_img = np.zeros((300, 700, 3), dtype=np.uint8)
    cv2.putText(_load_img, "VLM Model Loading...", (60, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 160, 255), 2)
    cv2.putText(_load_img, f"Environment: {env_profile.name}", (60, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1)
    cv2.putText(_load_img, "Please wait (10~30 sec)", (60, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 100), 1)
    cv2.namedWindow("HVAC Operator", cv2.WINDOW_NORMAL)
    cv2.imshow("HVAC Operator", _load_img)
    cv2.waitKey(1)

    vlm         = VLMProcessor()
    weather     = WeatherService(lat=WEATHER_LAT, lon=WEATHER_LON)
    air_quality = AirQualityService(service_key=AIR_QUALITY_API_KEY,
                                    station_name=AIR_QUALITY_STATION)
    hvac        = HVACSimulator(room_size=ROOM_SIZE_M2)
    hvac.set_room(ROOM_SIZE_M2, WINDOW_OPEN)
    engine      = ThermalEngine()
    sm          = StateManager(
        work_start_hour    = env_profile.work_start,
        work_end_hour      = env_profile.work_end,
        lunch_enabled      = env_profile.lunch_enabled,
        lunch_start        = env_profile.lunch_start,
        lunch_end          = env_profile.lunch_end,
        departure_enabled  = env_profile.departure_enabled,
    )
    motion_det  = MotionDetector(history_len=10, blur_ksize=21)
    yolo        = YOLODetector(imgsz=320, conf=0.35)
    pid         = PIDController(kp=0.8, ki=0.05, kd=0.3)
    sensor      = SensorInterface(simulator=hvac)

    # 더미 프레임 (카메라 없을 때 공통으로 사용)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Simulation Mode", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(dummy_frame, "No Camera Available", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # macOS: CAP_AVFOUNDATION 백엔드 사용 (권한 안정성)
    # Jetson CSI 카메라: nvarguscamerasrc GStreamer 파이프라인
    # Linux 일반: 기본 백엔드
    if platform.system() == "Darwin":
        import subprocess, re as _re
        cap = None
        # system_profiler로 카메라 목록 조회 → FaceTime/내장 카메라 인덱스 우선
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPCameraDataType"], timeout=5, text=True
            )
            # 내장 카메라가 리스트에서 몇 번째인지 파악
            names = _re.findall(r'^\s{4}(\S[^:]+):\s*$', out, _re.MULTILINE)
            # iPhone/아이폰이 아닌 첫 번째 카메라 = 내장 카메라
            builtin_idx = next(
                (i for i, n in enumerate(names) if "iPhone" not in n and "아이폰" not in n),
                0
            )
            cap = cv2.VideoCapture(builtin_idx, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = None
        except Exception:
            pass
        if cap is None:
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    elif _is_jetson():
        gst = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
            "nvvidconv flip-method=2 ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=1"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
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
    last_vlm_time      = None   # 마지막 VLM 분석 완료 시각 문자열
    vlm_analyzing      = False  # VLM 분석 중 여부
    out_temp, out_humid, out_weather, out_wind = 20.0, 50.0, "unknown", 0.0
    pm10, pm25, khai   = 0, 0, 0
    last_weather_fetch = 0.0
    last_pmv_update    = 0.0
    frame_count        = 0

    # ── 사용자 선호 + PMV 이력 ────────────────────────────────────────────────
    PMV_PREF_STEP  = 0.5
    PMV_PREF_MAX   = 2.0
    pmv_history: list = []
    PMV_HISTORY_MAX   = 30

    # 마우스 콜백을 위한 공유 상태 (mutable dict)
    pref_state = {'value': 0.0}

    def _user_mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicked = udisplay.get_clicked(x, y)
        if clicked == 'cold':   # 추워요 → 따뜻하게
            param['value'] = min(PMV_PREF_MAX,
                                 round(param['value'] + PMV_PREF_STEP, 1))
            print(f"[선호] 추워요 → 따뜻하게  오프셋={param['value']:+.1f}")
        elif clicked == 'hot':  # 더워요 → 시원하게
            param['value'] = max(-PMV_PREF_MAX,
                                 round(param['value'] - PMV_PREF_STEP, 1))
            print(f"[선호] 더워요 → 시원하게  오프셋={param['value']:+.1f}")

    # ── 재실 예측 ─────────────────────────────────────────────────────────────
    occ_pred              = {'message': '데이터 수집 중...', 'color': (100, 110, 145), 'record_count': 0}
    occ_pred_last_update  = 0.0
    OCC_PRED_INTERVAL_SEC = 60.0

    # ── 수동 제어 상태 ────────────────────────────────────────────────────────
    _key_q: queue.Queue = queue.Queue()

    def _on_press(key):
        try:
            _key_q.put(key.char)
        except AttributeError:
            pass

    if _PYNPUT_OK:
        _kb_listener = _kb.Listener(on_press=_on_press)
        _kb_listener.daemon = True
        _kb_listener.start()

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
                eff_met  = env_profile.met_baseline
                eff_clo  = _seasonal_clo(env_profile)
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

            # 상태 머신 업데이트
            if last_vlm_data is not None:
                sm.update(last_people_count,
                          last_vlm_data["outerwear"],
                          last_vlm_data["activity"])
            else:
                sm.update(last_people_count)

            # 사용자 선호 반영: adjusted_pmv로 제어
            adjusted_pmv = pmv_now - pref_state['value']
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
            last_vlm_time = datetime.now().strftime("%H:%M:%S")
            vlm_analyzing = False

            if not yolo.available:
                last_count_source = "vlm_fallback"

            log_row = process_vlm_result(
                vlm_data, last_people_count, last_count_source,
                motion_det, hvac, sm, engine, pid,
                sensor, display_state,
                out_temp, out_humid, out_weather, out_wind,
                pm10, pm25, khai,
                pmv_preference=pref_state['value'],
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
                             display_state, manual_ctrl, env_override,
                             vlm_data=last_vlm_data,
                             vlm_time=last_vlm_time,
                             vlm_analyzing=vlm_analyzing)
        panel_h = panel.shape[0]
        if cam_h < panel_h:
            pad        = np.zeros((panel_h - cam_h, display_frame.shape[1], 3), dtype=np.uint8)
            frame_disp = np.vstack([display_frame, pad])
        else:
            frame_disp = display_frame
        combined = np.hstack([frame_disp, panel])
        cv2.imshow("HVAC Operator", combined)

        # ── 사용자 인터페이스 창 ──────────────────────────────────────────────
        user_img = udisplay.build(
            hvac, sm, display_state,
            pref_state['value'], pmv_history, occ_pred, out_temp,
        )
        cv2.imshow("HVAC User", user_img)

        # 첫 프레임: 창 위치 분리
        if frame_count == 1:
            cv2.moveWindow("HVAC Operator", 0, 0)
            cv2.moveWindow("HVAC User", combined.shape[1] + 10, 0)
            cv2.setMouseCallback("HVAC User", _user_mouse_cb, pref_state)

        # ── 키 입력 처리 ──────────────────────────────────────────────────────
        cv_key = cv2.waitKey(1) & 0xFF
        try:
            ch = _key_q.get_nowait()
        except queue.Empty:
            # pynput 비활성(맥) 시 cv2.waitKey fallback
            ch = chr(cv_key) if cv_key not in (0, 255) else None

        if ch == "q":
            stop_event.set()
            vlm_thread.join(timeout=5)
            break

        elif ch == "w":
            hvac.window_open = not hvac.window_open
            print(f"[창문] {'열림' if hvac.window_open else '닫힘'}")

        elif ch == "u":
            pref_state['value'] = min(PMV_PREF_MAX,
                                      round(pref_state['value'] + PMV_PREF_STEP, 1))
            print(f"[선호] 따뜻하게  오프셋={pref_state['value']:+.1f}")

        elif ch == "d":
            pref_state['value'] = max(-PMV_PREF_MAX,
                                      round(pref_state['value'] - PMV_PREF_STEP, 1))
            print(f"[선호] 시원하게  오프셋={pref_state['value']:+.1f}")

        elif ch == "s":
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
                    pmv_preference=pref_state['value'],
                )
                save_log(log_row)

        # ── 수동 제어 키 (env_override 상태와 무관하게 항상 동작) ─────────────
        elif ch == "m":
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
            if ch == "p":
                manual_ctrl["power"] = not manual_ctrl["power"]
                print(f"[수동] 전원 {'ON' if manual_ctrl['power'] else 'OFF'}")
            elif ch == "c":
                manual_ctrl["mode"]  = "cool"
                manual_ctrl["power"] = True
                print("[수동] 냉방 모드")
            elif ch == "h":
                manual_ctrl["mode"]  = "heat"
                manual_ctrl["power"] = True
                print("[수동] 난방 모드")
            elif ch in ("=", "+"):
                manual_ctrl["target_temp"] = min(30.0, manual_ctrl["target_temp"] + 1.0)
                print(f"[수동] 설정온도 {manual_ctrl['target_temp']:.0f}°C")
            elif ch == "-":
                manual_ctrl["target_temp"] = max(16.0, manual_ctrl["target_temp"] - 1.0)
                print(f"[수동] 설정온도 {manual_ctrl['target_temp']:.0f}°C")
            elif ch == "f":
                manual_ctrl["fan_speed"] = manual_ctrl["fan_speed"] % 3 + 1
                print(f"[수동] 팬 속도 Fan {manual_ctrl['fan_speed']}")

        # ── 환경 오버라이드 키 ────────────────────────────────────────────────
        elif ch == "e":
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
            if ch == "[":
                env_override["selected"] = (env_override["selected"] - 1) % len(_ENV_VARS)
                print(f"[환경] 선택: {_ENV_LABEL[_ENV_VARS[env_override['selected']]]}")
            elif ch == "]":
                env_override["selected"] = (env_override["selected"] + 1) % len(_ENV_VARS)
                print(f"[환경] 선택: {_ENV_LABEL[_ENV_VARS[env_override['selected']]]}")
            elif ch in ("=", "+"):
                env_override[sel_key] = round(env_override[sel_key] + step, 1)
                print(f"[환경] {_ENV_LABEL[sel_key]} = {env_override[sel_key]}")
            elif ch == "-":
                env_override[sel_key] = round(env_override[sel_key] - step, 1)
                print(f"[환경] {_ENV_LABEL[sel_key]} = {env_override[sel_key]}")

    if use_camera:
        cap.release()
    cv2.destroyAllWindows()


def video_mode(video_path: str, analysis_interval: int, output_dir: str):
    """
    영상 파일 분석 모드 — 논문/발표용 데이터 생성.

    mp4/avi 등 영상 파일을 프레임별로 처리해 AI 제어 결과와
    룰베이스 베이스라인을 동시에 계산하고 CSV·그래프·요약을 자동 저장합니다.

    출력 구조 (output_dir/):
      analysis_log.csv          : 프레임별 전체 로그 (AI + 룰베이스)
      01_pmv_comparison.png     : PMV 시계열 비교
      02_indoor_temp.png        : 실내온도 비교
      03_energy_cumulative.png  : 누적 에너지 소비
      04_energy_bar.png         : 총 에너지 절감 막대
      05_comfort_rate.png       : 쾌적율 파이차트
      06_activity_distribution.png : VLM 활동 분포
      summary.txt               : 핵심 수치 요약
    """
    import report_generator

    if not os.path.exists(video_path):
        print(f"[Error] 영상 파일 없음: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "analysis_log.csv")

    # ── 컴포넌트 초기화 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  영상 분석 모드")
    print(f"  파일   : {video_path}")
    print(f"  출력   : {output_dir}")
    print(f"  VLM 주기: {analysis_interval}초")
    print(f"{'='*60}\n")

    # ── 로딩 화면 표시 ───────────────────────────────────────────────────────
    _loading_win = "HVAC Video Analysis"
    _load_img = np.zeros((300, 700, 3), dtype=np.uint8)
    cv2.putText(_load_img, "VLM Model Loading...", (60, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 160, 255), 2)
    cv2.putText(_load_img, os.path.basename(video_path), (60, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1)
    cv2.putText(_load_img, "Please wait (10~30 sec)", (60, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 100), 1)
    cv2.namedWindow(_loading_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_loading_win, 700, 300)
    cv2.imshow(_loading_win, _load_img)
    cv2.waitKey(1)

    vlm    = VLMProcessor()
    engine = ThermalEngine()
    pid_ai = PIDController(kp=0.8, ki=0.05, kd=0.3)
    pid_rb = PIDController(kp=0.8, ki=0.05, kd=0.3)
    yolo   = YOLODetector(imgsz=320, conf=0.35)
    hvac_ai = HVACSimulator(room_size=ROOM_SIZE_M2)
    hvac_rb = HVACSimulator(room_size=ROOM_SIZE_M2)  # 룰베이스 전용 시뮬레이터
    energy_ai_wh = 0.0
    energy_rb_wh = 0.0

    # ── 룰베이스 상수 (VLM 없이 고정값 사용) ────────────────────────────────
    RB_CLO      = 1.0    # 긴팔 기본
    RB_MET      = 1.2    # 기립 수준
    RB_SETPOINT = 24.0   # 고정 설정 온도 (°C)
    RB_FAN      = 2      # 재실 시 Fan2 고정

    POWER_W = {0: 0, 1: 800, 2: 1200, 3: 1600}

    # ── 영상 열기 ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    vlm_every_n  = max(1, int(fps * analysis_interval))

    print(f"  FPS: {fps:.1f}  |  총 {total_frames} 프레임  |  {duration_sec:.1f}초")
    print(f"  VLM 분석 주기: {vlm_every_n} 프레임마다 (≈{analysis_interval}초)\n")

    # ── CSV 헤더 ─────────────────────────────────────────────────────────────
    columns = [
        "video_time_sec", "frame_number",
        # VLM 출력
        "vlm_activity", "vlm_sleeves", "vlm_outerwear",
        "vlm_room_size", "vlm_heat_source", "vlm_raw",
        # 공통 입력
        "outdoor_temp", "outdoor_humid", "people_count",
        # AI 제어
        "ai_clo", "ai_met",
        "ai_pmv", "ai_comfort_status",
        "ai_hvac_on", "ai_target_temp", "ai_fan_speed", "ai_mode",
        "ai_indoor_temp", "ai_indoor_humid",
        "ai_energy_wh",
        # 룰베이스 베이스라인
        "rb_clo", "rb_met",
        "rb_pmv", "rb_comfort_status",
        "rb_hvac_on", "rb_target_temp", "rb_fan_speed",
        "rb_indoor_temp", "rb_indoor_humid",
        "rb_energy_wh",
    ]
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    # ── 상태 변수 ─────────────────────────────────────────────────────────────
    last_vlm_data   = None
    out_temp        = 20.0   # 외기 온도 (날씨 API 없이 고정, 추후 연동 가능)
    out_humid       = 50.0
    last_tick_time  = None   # 에너지 계산용
    frame_idx       = 0
    logged_rows     = 0

    hvac_ai.set_room(ROOM_SIZE_M2, False)
    hvac_rb.set_room(ROOM_SIZE_M2, False)

    print("분석 시작... (Ctrl+C로 중단 가능)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            video_time = frame_idx / fps

            # ── YOLO 인원 감지 (매 3초마다) ─────────────────────────────────
            if frame_idx % max(1, int(fps * 3)) == 0:
                people_count = yolo.count_people(frame) if yolo.available else 1
                people_count = max(0, people_count)

            # ── VLM 분석 (analysis_interval마다) ────────────────────────────
            if frame_idx % vlm_every_n == 0:
                print(f"  [VLM] {video_time:.1f}s / {duration_sec:.1f}s  "
                      f"({video_time/duration_sec*100:.0f}%)  "
                      f"인원: {people_count}명", flush=True)
                last_vlm_data = vlm.analyze_frame(frame)

            # VLM 결과 없으면 기본값
            if last_vlm_data is None:
                last_vlm_data = vlm._default_result()

            # ── 에너지 시간 계산 (1/fps 초) ──────────────────────────────────
            dt_h = (1.0 / fps) / 3600.0

            # ── AI 제어 계산 ─────────────────────────────────────────────────
            ai_clo = last_vlm_data["clo"]
            ai_met = last_vlm_data["met"]

            ai_in_t, ai_in_h = hvac_ai.indoor_temp, hvac_ai.indoor_humid
            ai_vel   = FAN_VELOCITY.get(hvac_ai.fan_speed, 0.1)
            ai_pmv   = engine.calculate_pmv(ai_in_t, ai_in_t, ai_in_h, ai_vel, ai_met, ai_clo)
            ai_comfort = engine.get_comfort_status(ai_pmv)

            ai_power, ai_tgt, ai_fan, ai_mode = decide_control(
                ai_pmv, people_count, pid_ai,
                hvac_ai.is_on, hvac_ai.mode,
                dt=1.0 / fps, current_fan=hvac_ai.fan_speed,
            )
            if ai_power is True:
                hvac_ai.set_control(power=True, target=ai_tgt, fan=ai_fan, mode=ai_mode)
            elif ai_power is False:
                hvac_ai.set_control(power=False, target=hvac_ai.target_temp, fan=1)
            hvac_ai.simulate_step(out_temp, out_humid, people_count)

            ai_watts      = POWER_W.get(hvac_ai.fan_speed, 0) if hvac_ai.is_on else 0
            energy_ai_wh += ai_watts * dt_h

            # ── 룰베이스 계산 ─────────────────────────────────────────────────
            rb_in_t, rb_in_h = hvac_rb.indoor_temp, hvac_rb.indoor_humid
            rb_vel   = FAN_VELOCITY.get(RB_FAN, 0.3)
            rb_pmv   = engine.calculate_pmv(rb_in_t, rb_in_t, rb_in_h, rb_vel, RB_MET, RB_CLO)
            rb_comfort = engine.get_comfort_status(rb_pmv)

            if people_count > 0:
                hvac_rb.set_control(power=True, target=RB_SETPOINT, fan=RB_FAN, mode="cool")
            else:
                hvac_rb.set_control(power=False, target=RB_SETPOINT, fan=1)
            hvac_rb.simulate_step(out_temp, out_humid, people_count)

            rb_watts      = POWER_W.get(RB_FAN, 0) if (people_count > 0) else 0
            energy_rb_wh += rb_watts * dt_h

            # ── 실시간 화면 표시 ─────────────────────────────────────────────
            disp = frame.copy()
            vt   = int(video_time)
            pct  = video_time / duration_sec * 100 if duration_sec > 0 else 0

            # 진행 바
            bar_w = int(disp.shape[1] * pct / 100)
            cv2.rectangle(disp, (0, disp.shape[0]-6), (disp.shape[1], disp.shape[0]), (40,40,40), -1)
            cv2.rectangle(disp, (0, disp.shape[0]-6), (bar_w, disp.shape[0]), (80,160,255), -1)

            # 오버레이 정보
            overlay_lines = [
                f"Video: {vt//60:02d}:{vt%60:02d} / {int(duration_sec)//60:02d}:{int(duration_sec)%60:02d}  ({pct:.0f}%)",
                f"People: {people_count}  |  AI PMV: {ai_pmv:+.2f}  |  RB PMV: {rb_pmv:+.2f}",
                f"AI Energy: {energy_ai_wh:.2f} Wh  |  RB Energy: {energy_rb_wh:.2f} Wh",
            ]
            if last_vlm_data and last_vlm_data.get('activity', '-') != '-':
                overlay_lines.append(
                    f"VLM: {last_vlm_data.get('activity','-')}  "
                    f"clo={last_vlm_data.get('clo',0):.2f}  "
                    f"met={last_vlm_data.get('met',0):.1f}"
                )
            for li, line in enumerate(overlay_lines):
                cv2.putText(disp, line, (10, 28 + li * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
                cv2.putText(disp, line, (10, 28 + li * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,80), 1)

            cv2.imshow("HVAC Video Analysis", disp)
            if frame_idx == 0:
                cv2.moveWindow("HVAC Video Analysis", 0, 0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ── VLM 분석 시점에만 CSV 로깅 (용량 절약) ───────────────────────
            if frame_idx % vlm_every_n == 0:
                row = {
                    "video_time_sec":  round(video_time, 2),
                    "frame_number":    frame_idx,
                    "vlm_activity":    last_vlm_data.get("activity", "-"),
                    "vlm_sleeves":     last_vlm_data.get("sleeves", "-"),
                    "vlm_outerwear":   last_vlm_data.get("outerwear", "-"),
                    "vlm_room_size":   last_vlm_data.get("room_size", "-"),
                    "vlm_heat_source": last_vlm_data.get("heat_source", "-"),
                    "vlm_raw":         str(last_vlm_data.get("raw_response", ""))[:200],
                    "outdoor_temp":    out_temp,
                    "outdoor_humid":   out_humid,
                    "people_count":    people_count,
                    "ai_clo":          ai_clo,
                    "ai_met":          ai_met,
                    "ai_pmv":          ai_pmv,
                    "ai_comfort_status": ai_comfort,
                    "ai_hvac_on":      hvac_ai.is_on,
                    "ai_target_temp":  hvac_ai.target_temp,
                    "ai_fan_speed":    hvac_ai.fan_speed,
                    "ai_mode":         hvac_ai.mode,
                    "ai_indoor_temp":  round(hvac_ai.indoor_temp, 2),
                    "ai_indoor_humid": round(hvac_ai.indoor_humid, 1),
                    "ai_energy_wh":    round(energy_ai_wh, 4),
                    "rb_clo":          RB_CLO,
                    "rb_met":          RB_MET,
                    "rb_pmv":          rb_pmv,
                    "rb_comfort_status": rb_comfort,
                    "rb_hvac_on":      people_count > 0,
                    "rb_target_temp":  RB_SETPOINT,
                    "rb_fan_speed":    RB_FAN if people_count > 0 else 0,
                    "rb_indoor_temp":  round(hvac_rb.indoor_temp, 2),
                    "rb_indoor_humid": round(hvac_rb.indoor_humid, 1),
                    "rb_energy_wh":    round(energy_rb_wh, 4),
                }
                pd.DataFrame([row]).to_csv(csv_path, mode="a", index=False, header=False)
                logged_rows += 1

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[중단됨] 지금까지 분석된 데이터로 리포트 생성합니다...")
    finally:
        cap.release()
        cv2.destroyWindow("HVAC Video Analysis")

    print(f"\n분석 완료: {frame_idx} 프레임 처리 / {logged_rows} 행 기록")
    print(f"CSV 저장: {csv_path}\n")

    # ── 리포트 생성 ───────────────────────────────────────────────────────────
    if logged_rows >= 2:
        report_generator.generate(csv_path, output_dir)
    else:
        print("[Report] 데이터 부족 — 최소 2개 VLM 분석 결과 필요")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM 기반 지능형 HVAC 제어 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  실시간 카메라:
    python main.py
    python main.py --interval 15

  영상 파일 분석 (논문/발표용):
    python main.py --video indoor.mp4
    python main.py --video indoor.mp4 --interval 10 --output results/my_test
        """,
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="VLM 분석 주기(초)  기본:30 / Mac:10~15 / Jetson:5~10",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        metavar="PATH",
        help="분석할 영상 파일 경로 (.mp4 / .avi 등). 지정 시 영상 분석 모드로 실행.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        metavar="DIR",
        help="결과 저장 폴더 (기본: results/video_분석시각/)",
    )
    args = parser.parse_args()

    if args.video:
        if args.output is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(args.video))[0]
            args.output = os.path.join("results", f"{base}_{ts}")
        video_mode(
            video_path=args.video,
            analysis_interval=args.interval,
            output_dir=args.output,
        )
    else:
        main(analysis_interval=args.interval)
