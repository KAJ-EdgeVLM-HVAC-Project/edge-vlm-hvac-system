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
from hvac_simulator import HVACSimulator
from thermal_engine import ThermalEngine
from state_machine import StateManager, SystemState
from motion_detector import MotionDetector
from yolo_detector import YOLODetector
from pid_controller import PIDController
from sensor_interface import SensorInterface
from control_logic import decide_control
from energy_monitor import EnergyMonitor, POWER_W, RB_SETPOINT, RB_FAN, rb_watts
from env_profiles import PROFILES, EnvProfile
from startup_screen import show_and_select, StartupResult
import dashboard as dash
import user_display as udisplay


def _is_youtube_url(path: str) -> bool:
    return 'youtube.com' in path or 'youtu.be' in path


def _download_youtube_temp(url: str) -> str | None:
    """YouTube 영상을 임시 파일로 다운로드. 경로 반환, 실패 시 None."""
    import yt_dlp, tempfile
    tmpdir = tempfile.mkdtemp(prefix="hvac_yt_")
    ydl_opts = {
        'format': 'best[ext=mp4]/best[height<=720]/best',
        'outtmpl': os.path.join(tmpdir, 'video.%(ext)s'),
        'quiet': False,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            ext  = info.get('ext', 'mp4')
        path = os.path.join(tmpdir, f'video.{ext}')
        return path if os.path.exists(path) else None
    except Exception as e:
        print(f"[YouTube] 다운로드 실패: {e}")
        return None

load_dotenv()


def _is_jetson() -> bool:
    return os.path.exists("/etc/nv_tegra_release")


LOG_FILE      = "hvac_system_performance.csv"
SCENARIO_NAME = "Smart_Office_Initial_Test"

WEATHER_API_KEY     = os.getenv("WEATHER_API_KEY", "")
WEATHER_LAT         = 35.1044
WEATHER_LON         = 128.9750
WEATHER_FETCH_SEC   = 60

ROOM_SIZE_M2    = 20.0
WORK_START_HOUR = 9
WORK_END_HOUR   = 18

# 디스플레이 없는 환경(SSH/systemd)에서 창 없이 구동
HEADLESS = (os.getenv("HVAC_HEADLESS") == "1" or
            (platform.system() == "Linux" and not os.getenv("DISPLAY")))

YOLO_EVERY_N_FRAMES = 90  # YOLO 인원 감지 주기 (3초마다 — 30fps 기준)
PMV_UPDATE_SEC      = 5   # PMV 재계산 + PID 제어 주기 (초)


# ── CSV 초기화 / 저장 ──────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "timestamp", "scenario", "system_state",
    "out_temp", "out_humid", "out_weather", "out_wind",
    "in_temp", "in_humid",
    "people_count", "count_source", "met", "clo", "activity",
    "heat_source", "motion_score", "met_source",
    "hvac_mode", "room_size",
    "pmv_val", "comfort_status", "target_temp", "fan_speed",
    "ai_energy_wh", "rb_energy_wh", "savings_pct", "comfort_rate",
]


def initialize_csv():
    # 기존 파일의 헤더가 현재 스키마와 다르면 백업 후 새로 생성
    if os.path.exists(LOG_FILE):
        try:
            old_header = pd.read_csv(LOG_FILE, nrows=0).columns.tolist()
        except Exception:
            old_header = None
        if old_header == CSV_COLUMNS:
            return
        backup = LOG_FILE.replace(
            ".csv", f"_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        os.rename(LOG_FILE, backup)
        print(f"[CSV] 스키마 변경 감지 — 기존 로그를 {backup} 로 백업")
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(LOG_FILE, index=False)


def save_log(data: dict):
    pd.DataFrame([data]).to_csv(LOG_FILE, mode="a", index=False, header=False)


# ── VLM 백그라운드 스레드 ──────────────────────────────────────────────────────

def vlm_worker(vlm, frame_lock, shared_frame_ref,
               result_queue, stop_event, interval, trigger_event=None,
               analyzing_event=None):
    """VLM을 백그라운드에서 주기적으로 실행 — 메인 루프를 블로킹하지 않음.
    trigger_event가 set되면 타이머 기다리지 않고 즉시 실행 (s키 수동 트리거용).
    analyzing_event는 추론 진행 중에만 set — 대시보드 '분석중...' 표시용.
    """
    while not stop_event.is_set():
        elapsed = 0.0
        while elapsed < interval and not stop_event.is_set():
            # trigger_event가 set되면 대기 중단하고 즉시 분석
            if trigger_event is not None and trigger_event.is_set():
                break
            time.sleep(0.5)
            elapsed += 0.5
        if stop_event.is_set():
            break
        if trigger_event is not None:
            trigger_event.clear()
        with frame_lock:
            if shared_frame_ref[0] is None:
                continue
            frame_copy = shared_frame_ref[0].copy()
        if analyzing_event is not None:
            analyzing_event.set()
        try:
            result = vlm.analyze_frame(frame_copy)
        finally:
            if analyzing_event is not None:
                analyzing_event.clear()
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
                       sensor, display_state, energy,
                       out_temp, out_humid, out_weather, out_wind,
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

    pmv_val     = engine.calculate_pmv(ta=sensor_temp, tr=tr_corrected,
                                       rh=sensor_humid, vel=0.1,
                                       met=effective_met, clo=vlm_data["clo"])
    comfort_msg = engine.get_comfort_status(pmv_val)

    # ── 상태 머신 업데이트 ────────────────────────────────────────────────────
    sm.update(people_count, vlm_data["outerwear"], vlm_data["activity"])

    hvac.set_room(vlm_data["room_size_m2"], False)
    energy.sync_room(vlm_data["room_size_m2"])

    # ── 제어 결정 (사용자 선호 + 시스템 상태 반영) ───────────────────────────
    # adjusted_pmv: 사용자가 따뜻함 선호(+) → 시스템이 더 춥다고 인식 → 난방 강화
    adjusted_pmv = pmv_val - pmv_preference
    power, target_temp, fan_speed, mode = decide_control(
        adjusted_pmv, people_count, pid, hvac.is_on, hvac.mode,
        current_fan=hvac.fan_speed, indoor_temp=hvac.indoor_temp,
        state=sm.state.value)

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
        "room_size":    hvac.room_size,
        "pmv_val":      pmv_val,
        "comfort_status": comfort_msg,
        "target_temp":  target_temp,
        "fan_speed":    fan_speed,
        "ai_energy_wh": round(energy.ai_wh, 2),
        "rb_energy_wh": round(energy.rb_wh, 2),
        "savings_pct":  round(energy.savings_pct, 1),
        "comfort_rate": round(energy.comfort_rate, 1),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main(analysis_interval: int = 30):
    # ── 시작 화면 (모드 + 환경 선택) ──────────────────────────────────────────
    if HEADLESS:
        # 디스플레이 없음 — 선택 화면 생략, 기본 프로파일로 카메라 모드 구동
        print("[Headless] DISPLAY 없음 — office 프로파일 / 카메라 모드로 시작")
        startup = StartupResult(mode='camera', video_path=None,
                                profile=PROFILES['office'])
    else:
        startup = show_and_select()

    # 영상 모드로 선택됐으면 video_mode로 넘김
    if startup.mode == 'video' and startup.video_path:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_to_delete = None

        if _is_youtube_url(startup.video_path):
            print(f"[YouTube] 임시 다운로드 중... (분석 후 자동 삭제)")
            print(f"  URL: {startup.video_path}")
            video_path = _download_youtube_temp(startup.video_path)
            if not video_path:
                print("[YouTube] 다운로드 실패. 종료합니다.")
                return
            tmp_to_delete = video_path
            out_dir = os.path.join("results", f"youtube_{ts}")
        else:
            video_path = startup.video_path
            base    = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join("results", f"{base}_{ts}")

        try:
            video_mode(video_path, analysis_interval, out_dir,
                       init_indoor_temp=startup.indoor_temp,
                       init_indoor_humid=startup.indoor_humid,
                       outdoor_temp=startup.outdoor_temp)
        finally:
            if tmp_to_delete and os.path.exists(tmp_to_delete):
                os.unlink(tmp_to_delete)
                tmpdir = os.path.dirname(tmp_to_delete)
                if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
                    os.rmdir(tmpdir)
                print("[YouTube] 임시 파일 삭제 완료")
        return

    env_profile = startup.profile

    initialize_csv()

    # ── 로딩 화면 ─────────────────────────────────────────────────────────────
    if not HEADLESS:
        _load_img = np.full((300, 700, 3), (249, 246, 244), dtype=np.uint8)  # BGR 라이트
        cv2.putText(_load_img, "VLM Model Loading...", (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 108, 79), 2)
        cv2.putText(_load_img, f"Environment: {env_profile.name}", (60, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (134, 116, 110), 1)
        cv2.putText(_load_img, "Please wait (10~30 sec)", (60, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (74, 163, 22), 1)
        cv2.namedWindow("HVAC Operator", cv2.WINDOW_NORMAL)
        cv2.imshow("HVAC Operator", _load_img)
        cv2.waitKey(1)

    vlm         = VLMProcessor()
    weather     = WeatherService(lat=WEATHER_LAT, lon=WEATHER_LON)
    hvac        = HVACSimulator(room_size=ROOM_SIZE_M2)
    hvac.set_room(ROOM_SIZE_M2, False)
    energy      = EnergyMonitor(room_size=ROOM_SIZE_M2,
                                init_temp=hvac.indoor_temp,
                                init_humid=hvac.indoor_humid)
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
    yolo        = YOLODetector(imgsz=640 if _is_jetson() else 320, conf=0.15)
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
    vlm_trigger      = threading.Event()   # s키 수동 트리거용
    vlm_analyzing_ev = threading.Event()   # 추론 진행 중 표시용

    vlm_thread = threading.Thread(
        target=vlm_worker,
        args=(vlm, frame_lock, shared_frame_ref,
              result_queue, stop_event, analysis_interval, vlm_trigger,
              vlm_analyzing_ev),
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
        "ai_wh": 0.0, "rb_wh": 0.0, "savings_pct": 0.0, "comfort_rate": 0.0,
    }

    last_people_count  = 0
    last_count_source  = "yolo"
    last_vlm_data      = None
    last_vlm_time      = None   # 마지막 VLM 분석 완료 시각 문자열
    out_temp, out_humid, out_weather, out_wind = 20.0, 50.0, "unknown", 0.0
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

    if _PYNPUT_OK and not HEADLESS:
        try:
            _kb_listener = _kb.Listener(on_press=_on_press)
            _kb_listener.daemon = True
            _kb_listener.start()
        except Exception as e:
            print(f"[키보드] pynput 리스너 시작 실패: {e} — cv2.waitKey 사용")

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

        # ── 날씨 갱신 (60초마다) ─────────────────────────────────────────────
        if time.time() - last_weather_fetch >= WEATHER_FETCH_SEC:
            out_temp, out_humid, out_weather, out_wind = weather.fetch_current_weather()
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
            pmv_now = engine.calculate_pmv(ta=s_temp, tr=tr_c,
                                           rh=s_humid, vel=0.1,
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

            # 쾌적율 누적 (재실 중만)
            energy.update_comfort(pmv_now, occupied=last_people_count > 0)

            # 사용자 선호 + 시스템 상태 반영: adjusted_pmv로 제어
            adjusted_pmv = pmv_now - pref_state['value']
            power, tgt, fan, mode = decide_control(
                adjusted_pmv, last_people_count, pid, hvac.is_on, hvac.mode,
                current_fan=hvac.fan_speed, indoor_temp=hvac.indoor_temp,
                state=sm.state.value)
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
                           people_count=last_people_count, dt=1/30)

        # 환경 오버라이드 시 시뮬 결과 덮어쓰기
        if env_override["enabled"]:
            hvac.indoor_temp  = env_override["indoor_temp"]
            hvac.indoor_humid = env_override["indoor_humid"]

        # ── 에너지 적산 (AI 실측 + 룰베이스 병렬 시뮬) ───────────────────────
        energy.step(1/30, eff_out_temp, eff_out_humid,
                    last_people_count, hvac)
        display_state["ai_wh"]        = energy.ai_wh
        display_state["rb_wh"]        = energy.rb_wh
        display_state["savings_pct"]  = energy.savings_pct
        display_state["comfort_rate"] = energy.comfort_rate

        # ── VLM 결과 처리 ─────────────────────────────────────────────────────
        try:
            vlm_data = result_queue.get_nowait()
            last_vlm_data = vlm_data
            last_vlm_time = datetime.now().strftime("%H:%M:%S")

            if not yolo.available:
                last_count_source = "vlm_fallback"

            log_row = process_vlm_result(
                vlm_data, last_people_count, last_count_source,
                motion_det, hvac, sm, engine, pid,
                sensor, display_state, energy,
                out_temp, out_humid, out_weather, out_wind,
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

        # ── 운영자 대시보드 + 사용자 창 렌더링 ──────────────────────────────
        if not HEADLESS:
            display_frame = cv2.resize(frame, (1280, 720)) if frame.shape[1] > 1280 else frame
            cam_h   = display_frame.shape[0]
            panel   = dash.build(cam_h, hvac, sm,
                                 out_temp, out_humid, out_weather, out_wind,
                                 display_state, manual_ctrl, env_override,
                                 vlm_data=last_vlm_data,
                                 vlm_time=last_vlm_time,
                                 vlm_analyzing=vlm_analyzing_ev.is_set())
            panel_h = panel.shape[0]
            if cam_h < panel_h:
                # 카메라 아래 여백을 패널 배경색(라이트)으로 채움
                pad        = np.full((panel_h - cam_h, display_frame.shape[1], 3),
                                     (249, 246, 244), dtype=np.uint8)
                frame_disp = np.vstack([display_frame, pad])
            else:
                frame_disp = display_frame
            combined = np.hstack([frame_disp, panel])
            cv2.imshow("HVAC Operator", combined)

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
        if HEADLESS:
            time.sleep(1/30)   # 창 없는 환경: waitKey 대신 루프 페이싱
            cv_key = 255
        else:
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

        elif ch == "u":
            pref_state['value'] = min(PMV_PREF_MAX,
                                      round(pref_state['value'] + PMV_PREF_STEP, 1))
            print(f"[선호] 따뜻하게  오프셋={pref_state['value']:+.1f}")

        elif ch == "d":
            pref_state['value'] = max(-PMV_PREF_MAX,
                                      round(pref_state['value'] - PMV_PREF_STEP, 1))
            print(f"[선호] 시원하게  오프셋={pref_state['value']:+.1f}")

        elif ch == "s":
            # VLM 워커 스레드에 즉시 실행 신호 — 메인 스레드에서 직접 호출 시 MPS segfault 발생
            vlm_trigger.set()
            print("[VLM] 수동 트리거 → 워커 스레드에서 즉시 분석")

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
    if not HEADLESS:
        cv2.destroyAllWindows()
    vlm.close()


def video_mode(video_path: str, analysis_interval: int, output_dir: str,
               init_indoor_temp: float = 26.0,
               init_indoor_humid: float = 50.0,
               outdoor_temp: float = 30.0):
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
    if not HEADLESS:
        _load_img = np.full((300, 700, 3), (249, 246, 244), dtype=np.uint8)  # BGR 라이트
        cv2.putText(_load_img, "VLM Model Loading...", (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 108, 79), 2)
        cv2.putText(_load_img, os.path.basename(video_path), (60, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (134, 116, 110), 1)
        cv2.putText(_load_img, "Please wait (10~30 sec)", (60, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (74, 163, 22), 1)
        cv2.namedWindow(_loading_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_loading_win, 700, 300)
        cv2.imshow(_loading_win, _load_img)
        cv2.waitKey(1)

    vlm    = VLMProcessor()
    engine = ThermalEngine()
    pid_ai = PIDController(kp=0.8, ki=0.05, kd=0.3)
    pid_rb = PIDController(kp=0.8, ki=0.05, kd=0.3)
    yolo   = YOLODetector(imgsz=640 if _is_jetson() else 320, conf=0.15)
    hvac_ai = HVACSimulator(room_size=ROOM_SIZE_M2)
    hvac_rb = HVACSimulator(room_size=ROOM_SIZE_M2)  # 룰베이스 전용 시뮬레이터
    energy_ai_wh = 0.0
    energy_rb_wh = 0.0

    # 룰베이스 상수(RB_SETPOINT/RB_FAN/POWER_W)는 energy_monitor 모듈과 공유

    # ── 영상 열기 ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    effective_interval = analysis_interval
    vlm_every_n = max(1, int(fps * effective_interval))

    print(f"  FPS: {fps:.1f}  |  총 {total_frames} 프레임  |  {duration_sec:.1f}초")
    print(f"  VLM 분석 주기: {vlm_every_n} 프레임마다 (≈{effective_interval}초)\n")

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
    last_panel      = None   # 매 프레임 재사용할 정보 패널
    ai_pmv          = 0.0
    rb_pmv          = 0.0
    vlm_step        = 0
    out_temp        = outdoor_temp
    out_humid       = 60.0
    people_count    = 0
    logged_rows     = 0

    # 초기 실내 조건 설정
    hvac_ai.set_room(ROOM_SIZE_M2, False)
    hvac_rb.set_room(ROOM_SIZE_M2, False)
    hvac_ai.indoor_temp  = init_indoor_temp
    hvac_ai.indoor_humid = init_indoor_humid
    hvac_rb.indoor_temp  = init_indoor_temp
    hvac_rb.indoor_humid = init_indoor_humid

    yolo_every_n = max(1, int(fps * 3))   # YOLO: 3초마다
    vlm_every_n  = max(1, int(fps * effective_interval))  # VLM: interval마다

    total_steps  = total_frames // vlm_every_n
    frame_count  = 0

    print(f"  초기 실내: {init_indoor_temp:.1f}°C / {init_indoor_humid:.1f}%  외기: {out_temp:.1f}°C")
    print(f"  YOLO: 3초마다  VLM: {effective_interval}초마다  총 예상 분석: {total_steps}회\n")
    print("분석 시작... (q키로 중단 가능)\n")

    if not HEADLESS:
        if _is_jetson():
            cv2.startWindowThread()
        cv2.namedWindow("HVAC Video Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HVAC Video Analysis", 960, 540)
        cv2.moveWindow("HVAC Video Analysis", 0, 0)
        cv2.waitKey(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            video_time   = frame_count / fps
            pct          = min(frame_count / total_frames * 100, 100.0)

            # ── YOLO: 3초마다 ────────────────────────────────────────────────
            if frame_count % yolo_every_n == 0 and yolo.available:
                people_count = max(0, yolo.count_people(frame))

            # ── VLM 주기 아닌 프레임: 패널 재사용하여 표시 ─────────────────
            if frame_count % vlm_every_n != 0:
                if not HEADLESS:
                    if last_panel is not None:
                        # 진행 바만 업데이트
                        bar_w = int(frame.shape[1] * pct / 100)
                        cur_panel = last_panel.copy()
                        cv2.rectangle(cur_panel, (0, 0), (frame.shape[1], 6), (40,40,50), -1)
                        cv2.rectangle(cur_panel, (0, 0), (bar_w, 6), (80,160,255), -1)
                        disp = np.vstack([frame, cur_panel])
                    else:
                        disp = frame
                    cv2.imshow("HVAC Video Analysis", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            vlm_step += 1
            dt_sec = effective_interval
            dt_h   = dt_sec / 3600.0

            if people_count > 0:
                print(f"  [{vlm_step}] {video_time:.0f}s ({pct:.0f}%)  "
                      f"인원:{people_count}  VLM 분석 중...", flush=True)
                last_vlm_data = vlm.analyze_frame(frame) or vlm._default_result()
            else:
                print(f"  [{vlm_step}] {video_time:.0f}s ({pct:.0f}%)  "
                      f"인원:0  VLM 스킵 (사람 없음)", flush=True)
                last_vlm_data = {
                    "sleeves": "-", "clo": 1.0, "met": 1.0,
                    "room_size": "medium", "room_size_m2": ROOM_SIZE_M2,
                    "heat_source": "no", "outerwear": "no", "activity": "-",
                }

            # ── 물리 시뮬레이션 누적 ─────────────────────────────────────────
            sim_steps   = max(1, int(dt_sec * 10))   # 0.1초 단위 분할
            sim_dt      = dt_sec / sim_steps
            for _ in range(sim_steps):
                hvac_ai.simulate_step(out_temp, out_humid, people_count, dt=sim_dt)
                hvac_rb.simulate_step(out_temp, out_humid, people_count, dt=sim_dt)

            # ── AI 제어 계산 ─────────────────────────────────────────────────
            ai_clo = last_vlm_data["clo"]
            ai_met = last_vlm_data["met"]
            # heat_source → 복사온도 보정 (카메라 모드와 동일 로직)
            ai_tr = hvac_ai.indoor_temp
            if last_vlm_data.get("heat_source") == "yes":
                ai_tr += VLMProcessor.TR_HEAT_OFFSET
            # room_size → 같은 물리 공간이므로 AI/RB 시뮬레이터 모두 갱신
            detected_room_m2 = last_vlm_data.get("room_size_m2", ROOM_SIZE_M2)
            hvac_ai.set_room(detected_room_m2, False)
            hvac_rb.set_room(detected_room_m2, False)

            if people_count > 0:
                # PMV 기류속도: 팬속도는 온도 시뮬레이션에만 사용.
                # 실내 체감 기류는 사무실 정지공기 기준 0.1 m/s 고정 (ISO 7730)
                PMV_VEL = 0.1
                ai_pmv = engine.calculate_pmv(
                    hvac_ai.indoor_temp, ai_tr,
                    hvac_ai.indoor_humid, PMV_VEL, ai_met, ai_clo)
                ai_comfort = engine.get_comfort_status(ai_pmv)
            else:
                ai_pmv    = None
                ai_comfort = "-"

            ai_power, ai_tgt, ai_fan, ai_mode = decide_control(
                ai_pmv if ai_pmv is not None else 0.0, people_count, pid_ai,
                hvac_ai.is_on, hvac_ai.mode,
                dt=dt_sec or 1.0, current_fan=hvac_ai.fan_speed,
                indoor_temp=hvac_ai.indoor_temp)
            if ai_power is True:
                hvac_ai.set_control(power=True, target=ai_tgt, fan=ai_fan, mode=ai_mode)
            elif ai_power is False:
                hvac_ai.set_control(power=False, target=hvac_ai.target_temp, fan=1)

            ai_watts      = POWER_W.get(hvac_ai.fan_speed, 0) if hvac_ai.is_on else 0
            energy_ai_wh += ai_watts * dt_h

            # ── 룰베이스 계산 ─────────────────────────────────────────────────
            # PMV 측정은 실제 재실자 기준(VLM 감지값)으로 통일 — RB 제어만 고정 24°C/Fan2
            if people_count > 0:
                rb_pmv = engine.calculate_pmv(
                    hvac_rb.indoor_temp, hvac_rb.indoor_temp,
                    hvac_rb.indoor_humid, PMV_VEL, ai_met, ai_clo)
                rb_comfort = engine.get_comfort_status(rb_pmv)
            else:
                rb_pmv    = None
                rb_comfort = "-"

            if people_count > 0:
                rb_mode = "heat" if hvac_rb.indoor_temp < RB_SETPOINT else "cool"
                hvac_rb.set_control(power=True, target=RB_SETPOINT, fan=RB_FAN, mode=rb_mode)
            else:
                hvac_rb.set_control(power=False, target=RB_SETPOINT, fan=1)

            # RB 에너지: 설정온도 도달 후 서모스탯 사이클링(간헐 작동) 반영
            # 실제 온도조절기는 설정온도 ±0.5°C 이내에서 컴프레서 정지/팬만 운전
            # → 설정온도에서는 25% 소비(팬+대기), 적극 가열·냉방 시 100% 소비
            energy_rb_wh += rb_watts(hvac_rb.indoor_temp, people_count) * dt_h

            # ── 화면 표시 ────────────────────────────────────────────────────
            disp  = frame.copy()
            h_f, w_f = disp.shape[:2]
            vt    = int(video_time)

            # 하단 정보 패널 (라이트 배경)
            panel_h = 160
            panel = np.full((panel_h, w_f, 3), (249, 246, 244), dtype=np.uint8)
            cv2.line(panel, (0, 7), (w_f, 7), (236, 231, 228), 1)

            # 진행 바
            bar_w = int(w_f * pct / 100)
            cv2.rectangle(panel, (0, 0), (w_f, 6), (236, 231, 228), -1)
            cv2.rectangle(panel, (0, 0), (bar_w, 6), (255, 108, 79), -1)

            ai_on_str  = f"ON  Tgt:{hvac_ai.target_temp:.0f}C Fan{hvac_ai.fan_speed}" if hvac_ai.is_on  else "OFF"
            rb_on_str  = f"ON  Tgt:{RB_SETPOINT:.0f}C Fan{RB_FAN}"                   if people_count>0 else "OFF"

            rows = [
                # 시간 / 진행률
                (f"[{vlm_step}] {vt//60:02d}:{vt%60:02d} / "
                 f"{int(duration_sec)//60:02d}:{int(duration_sec)%60:02d}  ({pct:.0f}%)"
                 f"   People:{people_count}   Outdoor:{out_temp:.1f}C",
                 (134, 124, 120)),
                # AI 제어 상태 (블루)
                (f"[AI]  실내:{hvac_ai.indoor_temp:.1f}C  습도:{hvac_ai.indoor_humid:.0f}%"
                 f"  PMV:{'--' if ai_pmv is None else f'{ai_pmv:+.2f}'}  {ai_on_str}  {energy_ai_wh:.1f}Wh",
                 (235, 118, 37)),
                # 룰베이스 상태 (오렌지)
                (f"[RB]  실내:{hvac_rb.indoor_temp:.1f}C  습도:{hvac_rb.indoor_humid:.0f}%"
                 f"  PMV:{'--' if rb_pmv is None else f'{rb_pmv:+.2f}'}  {rb_on_str}  {energy_rb_wh:.1f}Wh",
                 (10, 138, 228)),
                # VLM 분석 결과 (그린)
                (f"[VLM] {last_vlm_data.get('activity','-')}"
                 f"  sleeves:{last_vlm_data.get('sleeves','-')}"
                 f"  clo:{last_vlm_data.get('clo',0):.2f}"
                 f"  met:{last_vlm_data.get('met',0):.1f}"
                 f"   Saved:{energy_rb_wh-energy_ai_wh:.1f}Wh",
                 (74, 163, 22)),
            ]
            for ri, (text, color) in enumerate(rows):
                y = 26 + ri * 32
                cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
                            cv2.LINE_AA)

            last_panel = panel
            if not HEADLESS:
                disp = np.vstack([disp, panel])
                cv2.imshow("HVAC Video Analysis", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ── CSV 로깅 ─────────────────────────────────────────────────────
            if True:
                row = {
                    "video_time_sec":  round(video_time, 2),
                    "frame_number":    frame_count,
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
                    "rb_clo":          ai_clo,
                    "rb_met":          ai_met,
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

    except KeyboardInterrupt:
        print("\n[중단됨] 지금까지 분석된 데이터로 리포트 생성합니다...")
    finally:
        cap.release()
        if not HEADLESS:
            cv2.destroyWindow("HVAC Video Analysis")
        vlm.close()

    print(f"\n분석 완료: {vlm_step} 포인트 분석  ({logged_rows} CSV 행)")
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
