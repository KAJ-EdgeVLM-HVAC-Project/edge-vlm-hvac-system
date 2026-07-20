# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Edge-deployed VLM-based intelligent HVAC control system. A camera analyzes occupants' clothing, posture, and activity via a Vision Language Model (Qwen3-VL-2B / Qwen2-VL-2B), computes PMV (ISO 7730:2005), and controls a simulated HVAC unit accordingly. Designed for Jetson Orin Nano Super deployment with Mac development support.

## Running the System

**Mac (development):**
```bash
source .venv/bin/activate
python main.py                  # 30s VLM interval (default)
python main.py --interval 10    # 10s interval (M-series Mac)
```

**Jetson Orin Nano Super (172.20.10.11, user: jetson):**
```bash
cd ~/edge-vlm-hvac-system
./run.sh                        # venv/DISPLAY auto-detect, GGML_CUDA_NO_VMM=1
# headless 강제: HVAC_HEADLESS=1 ./run.sh
# 상시 가동: deploy/hvac.service (systemd, headless) 참고
```

DISPLAY가 없으면 자동으로 headless 모드(창 없이 콘솔+CSV만)로 동작한다.

**Environment setup (.env):**
```
WEATHER_API_KEY=...   # 기상청 초단기실황 (data.go.kr)
```

## Architecture

### Data Flow
```
Camera frame
  → YOLODetector      : bounding boxes (Jetson: YOLO26s TensorRT GPU ~39ms)
  → OccupancyTracker  : people count (이동예측 + 헝가리안 매칭, 출입구 자동 학습)
  → VLMProcessor      : activity, clo, met, room_size, heat_source
  → ThermalEngine     : PMV/PPD (ISO 7730:2005)
  → StateManager      : EMPTY/ARRIVAL/STEADY/PRE_DEPARTURE/LUNCH_BREAK
  → decide_control()  : power, target_temp, fan_speed, mode
                        (state 파라미터로 ARRIVAL 부스트/PRE_DEPARTURE 절전/
                         LUNCH_BREAK 약운전 반영)
  → HVACSimulator     : simulates indoor_temp, indoor_humid per frame
  → EnergyMonitor     : AI vs 룰베이스(24°C/Fan2) 실시간 Wh 비교 + 쾌적율
  → Dashboard + UserDisplay : two OpenCV windows (headless 시 생략)
  → CSV log (hvac_system_performance.csv)
```

### Key Modules

| File | Role |
|------|------|
| `main.py` | Orchestration, camera loop, threading, CSV logging, video mode |
| `vlm_processor.py` | VLM inference with auto device selection (lcpp→mps→cuda→cpu) |
| `thermal_engine.py` | PMV/PPD calculation (ISO 7730:2005) |
| `control_logic.py` | PMV → HVAC decision (dynamic target temp + PID fan + state-aware) |
| `state_machine.py` | Occupancy state transitions with departure/lunch detection |
| `hvac_simulator.py` | Physical indoor environment simulation |
| `energy_monitor.py` | AI vs rule-based baseline energy comparison (live mode) |
| `yolo_detector.py` | YOLO26s people detection (Jetson: torch-free TensorRT 엔진) |
| `occupancy_tracker.py` | 재실 인원 추적 — 이동예측 + 헝가리안 매칭, 출입구 자동 학습 |
| `startup_screen.py` | Environment profile selector (shown on launch) |
| `dashboard.py` | Operator OpenCV window rendering |
| `user_display.py` | User-facing OpenCV window rendering |
| `sensor_interface.py` | SHT31 I2C temperature/humidity sensor (GPIO pins 3/5) |
| `env_profiles.py` | Environment presets (office, home, gym, etc.) |
| `scenario_runner.py` | Offline scenario simulation from JSON files |
| `report_generator.py` | Video-mode analysis report (graphs + summary) |

### VLM Device Priority
`vlm_processor.py` auto-selects backend at startup:
1. **lcpp** — llama.cpp CUDA INT4 (Jetson). `llama-server` 상주 프로세스를 우선
   기동(모델 1회 로드, HTTP + json_schema 강제)하고, 실패 시
   `llama-mtmd-cli` subprocess(매 추론마다 모델 재로딩)로 폴백.
2. **mps** — Apple Silicon MPS (HuggingFace Qwen2-VL-2B)
3. **cuda** — NVIDIA CUDA (HuggingFace)
4. **cpu** — CPU fallback

**Jetson GGUF 모델 자동 탐색:** `~/llama.cpp/models/*/` 에서 (모델.gguf +
mmproj*.gguf) 쌍을 찾으며, 디렉토리명에 `qwen3` 포함 시 우선 사용.
`LCPP_GGUF` / `LCPP_MMPROJ` 환경변수로 직접 지정 가능.

권장 모델: **Qwen3-VL-2B-Instruct GGUF Q4_K_M** (공식:
huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) — `~/llama.cpp/models/Qwen3-VL-2B/`
에 모델+mmproj를 받아두면 자동 선택됨. 기존 Qwen2-VL-2B(Q4_K_M, 941MB)는 폴백.

### PMV 정확도 관련 주의사항 (`thermal_engine.py`, `vlm_processor.py`)

과거에 "한여름 실내 27°C인데 쾌적 판정" 문제가 있었고, 원인은 셋이었다.
아래 값을 임의로 되돌리면 같은 증상이 재발한다.

1. **수증기 분압 단위** — ISO 7730 Annex D는 `pa = RH[%] × 10 × exp(...)` 로 **Pa**
   단위다. `rh/100`으로 쓰면 100배 작아져 증발 열손실이 과대평가되고, 습도가
   PMV에 거의 반영되지 않는다.
2. **대사율(met)** — `sitting = 1.2` (ISO 7730 Table B.1 *sedentary office work*).
   1.0은 *seated, relaxed*(눈 감고 휴식) 수준이라 사무·연구실 환경에 맞지 않는다.
   1.0을 쓰면 냉방 시작이 27.3°C까지 밀린다 (1.2면 26.0°C).
3. **기류 속도** — 냉난방 가동 중에는 `FAN_VEL`(0.10~0.25 m/s)로 팬 단계를 반영한다.
   정지공기 고정은 "냉방 중인데 정지공기" 모순을 만든다. 단, 값을 크게 잡으면
   켜자마자 PMV가 급락해 on/off 진동이 생긴다.

**외기 적응 보정**(`adaptive_pmv_offset`)은 목표 PMV 기준선만 ±0.4 이동시킨다.
ASHRAE 55 적응형 공식(`0.31 × 외기 + 17.8`)을 목표온도로 직접 쓰지 않는 이유는,
그 공식이 **자연환기 건물** 대상이라 외기 31°C에서 실내 27.4°C를 쾌적으로 보기
때문이다. ASHRAE 55도 공조 건물에는 PMV를 쓰라고 명시한다.

### Energy Estimation Model
`energy_monitor.py` — AI 제어 vs 룰베이스 비교 (camera/video 모드 공통, `hvac_watts()`):
- 압축기 부하 기반: `P = P_fan + P_comp,rated × load`
- 팬 1/2/3: 40/70/110 W, 압축기 정격: 1200 W
- 부하율 `load` = 실내로 새어드는 열(외기 전도 + 체열) ÷ 에어컨 냉난방 능력
  → 설정온도까지 구동 시 1.0(정격), 유지 시 열손실 상쇄분만큼 부분부하(사이클링).
  부하율 상수(τ·냉난방률·체열)는 `hvac_simulator.py`와 일치시킴 → 설정온도·외기 반영
- 룰베이스: 재실 시 계절별 고정(냉방 24°C / 난방 25°C, 외기 20°C 기준) + Fan2
- Comfort rate: 재실 중 PMV ∈ (-0.5, 0.5) 프레임 비율

### CSV Log Schema (`hvac_system_performance.csv`)
Key columns: `timestamp, system_state, people_count, activity, clo, met, pmv_val,
in_temp, in_humid, out_temp, hvac_mode, fan_speed, target_temp,
ai_energy_wh, rb_energy_wh, savings_pct, comfort_rate`
(스키마 변경 시 기존 파일은 자동 백업 후 재생성)

## Jetson-Specific Notes

**Critical env vars for Jetson:**
- `GGML_CUDA_NO_VMM=1` — prevents CUDA OOM on unified memory (run.sh가 설정)
- `HVAC_HEADLESS=1` — 강제 headless (DISPLAY 없으면 자동)
- 구 보드(hanul/JetPack 5.1.2)만 LD_PRELOAD(libgomp+libGLdispatch) 필요했음.
  새 보드(jetson/JetPack 6.x)는 불필요.

**cv2 on Jetson:** PyPI opencv (Qt backend) causes NULL window handler crash. Use system GTK opencv:
```bash
ln -sfn /usr/lib/python3/dist-packages/cv2.cpython-310-aarch64-linux-gnu.so \
    ~/edge-vlm-hvac-system/vlm-env/lib/python3.10/site-packages/cv2.so
```

**pynput on Mac:** Causes HIToolbox crash on macOS 26+. Disabled via `_PYNPUT_OK = platform.system() != "Darwin"`. On Mac, keyboard input falls back to `cv2.waitKey()`.

**llama.cpp build (Jetson Orin Nano Super, CUDA arch 87):**
```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6
cmake --build build --config Release -j4
```
`llama-server` 타깃도 함께 빌드됨 (`build/bin/llama-server`) — 상주 추론에 필요.

## Branches

- `main` — **현재 활성**. Mac/Jetson 공용. llama-server 상주, 상태머신 제어 연동,
  실시간 에너지 모니터, headless, YOLO26s TensorRT GPU, 재실 추적 고도화까지 병합 완료
- `feature/llamacpp-int4-quantization` — Jetson llama.cpp CUDA INT4 (구버전, 보관용)
- `feature/final-overhaul` — main에 병합 완료 (보관용)

## Removed Features (의도적 제거)

- 공기질(PM10/PM2.5) API 연동 — 실내 공조 제어에 집중하기 위해 제거
- 창문 개폐 권장(decide_window) — 라이브 시스템에서 제거
  (scenario_runner.py의 오프라인 시뮬레이션에는 잔존)
- TensorRT-LLM 백엔드 — Orin Nano에서 비실용적, llama.cpp로 대체
- `week2/`~`week8/` 폴더(타 교과목 제출물) — 제거됨
