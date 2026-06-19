# 엣지 VLM 기반 지능형 HVAC 제어 시스템

**Edge VLM-Driven On-Device AI Intelligent HVAC Control System**

카메라 영상에서 재실자의 착의량·활동량을 Vision Language Model(VLM)로 실시간 분석하고,
ISO 7730:2005 PMV 열쾌적 지수를 계산해 공조기를 자동 제어하는 엣지 온디바이스 AI 시스템.

> 동아대학교 컴퓨터공학과 캡스톤디자인 — 김엔정 조 (김준경·김철호·김민서·정윤찬)

📹 **데모 영상:** [docs/demo/vlm_hvac_demo.mp4](docs/demo/vlm_hvac_demo.mp4) — 동일 인물이 민소매→반팔→아우터로 옷을 바꿀 때 clo·PMV·냉난방이 자동으로 적응하는 실시간 시연
📊 **실험 결과:** [docs/EXPERIMENT.md](docs/EXPERIMENT.md) — 실제 사무실 영상(Edinburgh Office Dataset) 분석, 데이터셋 출처·그래프·원자료 포함

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [아키텍처](#아키텍처)
3. [모듈 구성](#모듈-구성)
4. [VLM 백엔드 선택](#vlm-백엔드-선택)
5. [실행 방법](#실행-방법)
6. [환경 설정](#환경-설정)
7. [영상 분석 모드](#영상-분석-모드)
8. [Jetson 배포](#jetson-orin-nano-super-배포)
9. [실험 결과](#실험-결과)
10. [브랜치 안내](#브랜치-안내)

---

## 시스템 개요

기존 규칙 기반 HVAC는 온도 센서 수치 하나에만 의존해 재실자의 착의량·활동량·인원 변화를 반영하지 못한다. 본 시스템은 다음 3단계 파이프라인으로 이 한계를 극복한다.

| 단계 | 내용 |
|------|------|
| **인지 (Perception)** | YOLOv8n으로 인원 카운팅 / Qwen3-VL-2B(GGUF)로 착의·활동 맥락 파악 / MotionDetector로 실시간 MET 보정 |
| **판단 (Context)** | ISO 7730:2005 PMV 6변수 계산 / 5단계 상태 머신(공실·도착·안정·점심·퇴근) |
| **제어 (Control)** | PID + 동적 목표온도 + 히스테리시스 / RC 열회로 물리 시뮬레이션 |

**모든 추론은 온디바이스(Jetson Orin Nano Super)에서 수행 — 카메라 영상 외부 전송 없음.**

---

## 아키텍처

```
카메라 프레임 (30fps)
  │
  ├─ [매 프레임]    MotionDetector   → motion_score → MET 실시간 보정
  ├─ [매 3초]       YOLODetector     → people_count
  ├─ [매 30초/bg]   VLMProcessor     → clo / met / room_size / heat_source
  ├─ [매 60초]      WeatherService   → outdoor_temp / humid
  │
  └─ [매 5초] 제어 루프
        ThermalEngine  : PMV/PPD 계산 (ISO 7730:2005)
        StateManager   : EMPTY → ARRIVAL → STEADY ⇄ LUNCH_BREAK / PRE_DEPARTURE
        decide_control : power / target_temp / fan_speed / mode
                         (ARRIVAL 부스트 · PRE_DEPARTURE 절전 · LUNCH 약운전)
        HVACSimulator  : indoor_temp / indoor_humid 물리 시뮬레이션 (RC 열회로 τ=3600s)
        EnergyMonitor  : AI vs 룰베이스(24°C/Fan2) 실시간 Wh 비교 + 쾌적율
        │
        ├─ Dashboard    : 운영자 창 (카메라 + 상태 패널)   ※ headless 시 생략
        ├─ UserDisplay  : 사용자 창 (리모컨 UI)
        └─ CSV 로그     : hvac_system_performance.csv
```

### PMV 입력 6변수

| 변수 | 소스 | 비고 |
|------|------|------|
| `ta` 공기온도 | SHT31 센서 / 시뮬레이션 | °C |
| `tr` 복사온도 | ta + 열원보정 (+4°C) | 조리기구 감지 시 적용 |
| `rh` 상대습도 | SHT31 센서 / 시뮬레이션 | % |
| `vel` 기류속도 | **고정 0.1 m/s** | ISO 7730 정지기류 기준 (팬 속도 미반영) |
| `met` 대사율 | VLMProcessor / MotionDetector | 1.0(착석) ~ 3.0(운동) |
| `clo` 착의량 | VLMProcessor / 외부온도 기반 fallback | 0.5(반팔) ~ 1.3(아우터) |

### 제어 전략 (AI vs 규칙기반)

| 항목 | AI 제어 | 규칙기반 |
|------|---------|---------|
| 목표온도 | PMV 기반 동적 (18~30°C) | 24°C 고정 |
| 최소온도 보장 | 22°C (재실 시) | 해당 없음 |
| 팬 속도 | PID 비례 (1~3단) | Fan2 고정 |
| 공실 감지 | 즉시 OFF | 없음 |
| 점심·퇴근 예측 | 5단계 상태 머신 | 없음 |

---

## 모듈 구성

```
main.py               메인 루프 — 카메라·영상 모드, 스레딩, CSV 로그
vlm_processor.py      VLM 추론 — 백엔드 자동 선택 (lcpp→mps→cuda→cpu)
thermal_engine.py     PMV/PPD 계산 (ISO 7730:2005 완전 구현)
control_logic.py      PMV → HVAC 제어 결정 (PID + 동적목표온도 + 상태머신 연동)
energy_monitor.py     AI vs 룰베이스 에너지 비교 (실시간 + 영상 모드 공통)
hvac_simulator.py     RC 열회로 물리 시뮬레이션 (τ=3600s, dt 기반 통일 물리)
state_machine.py      5단계 재실 상태 전이 (맥락 점수 기반 퇴근 예측)
pid_controller.py     PID 제어기 (anti-windup, deadband 포함)
yolo_detector.py      YOLOv8n 인원 카운팅 (imgsz=320 Mac / 640 Jetson)
motion_detector.py    프레임 차분 기반 움직임 강도 → MET 변환
sensor_interface.py   SHT31 I2C 온습도 센서 (Jetson GPIO Pin3/5, Bus7, 0x44)
weather_service.py    기상청 API — 외기온도·습도·날씨
env_profiles.py       환경 프로파일 (사무실/가정/체육시설/부대시설)
startup_screen.py     시작화면 — 환경·카메라·동작 모드 선택 GUI
dashboard.py          운영자 대시보드 OpenCV 렌더링
user_display.py       사용자 리모컨 UI OpenCV 렌더링
report_generator.py   영상 분석 완료 후 논문용 그래프·통계 자동 생성
scenario_runner.py    JSON 시나리오 기반 오프라인 시뮬레이션 (독립 실행)
```

---

## VLM 백엔드 선택

`VLMProcessor`가 시작 시 자동으로 최적 백엔드를 선택한다.

| 우선순위 | 백엔드 | 조건 | 비고 |
|----------|--------|------|------|
| 1 | **lcpp** — llama.cpp CUDA INT4 | `~/llama.cpp/build` CUDA 빌드 + GGUF 존재 | Jetson GPU |
| 2 | **mps** — Apple Silicon | macOS + MPS 사용 가능 | Mac 개발용 |
| 3 | **cuda** — HuggingFace CUDA | CUDA 사용 가능 | |
| 4 | **cpu** — fallback | 항상 | 느림 |

**lcpp 추론 방식 (Jetson):**
1. **llama-server 상주** (기본) — 시작 시 모델 1회 로드 후 HTTP로 추론.
   `response_format: json_schema`로 출력 JSON을 문법 수준에서 강제 → 파싱 실패 원천 차단.
2. **llama-mtmd-cli 폴백** — 서버 기동 실패 시. 매 추론마다 모델 재로딩되어 느림.

**Jetson GGUF 모델 자동 탐색:** `~/llama.cpp/models/*/` 에서 (모델.gguf + mmproj.gguf)
쌍을 찾고, 디렉토리명에 `qwen3` 포함 시 우선 사용. `LCPP_GGUF`/`LCPP_MMPROJ` 환경변수로
직접 지정 가능.

```
# 권장 (Qwen3-VL-2B — huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF)
~/llama.cpp/models/Qwen3-VL-2B/Qwen3-VL-2B-Instruct-Q4_K_M.gguf
~/llama.cpp/models/Qwen3-VL-2B/mmproj-Qwen3-VL-2B-Instruct-F16.gguf

# 폴백 (기존 Qwen2-VL-2B)
~/llama.cpp/models/Qwen2-VL-2B/qwen2vl-2b-q4km.gguf          # Q4_K_M (941 MB)
~/llama.cpp/models/Qwen2-VL-2B/mmproj-qwen2vl-2b-f16.gguf    # FP16 (1.3 GB)
```

**VLM 파싱 방어 구조 (CLI 폴백 시):**
1. ggml 내부 로그 라인 제거
2. 타이밍 줄(숫자로 시작) 제거
3. CUDA 에러 키워드 라인 제거
4. JSON 추출 실패 시 → 개별 필드 Regex fallback

---

## 실행 방법

### Mac (개발 환경)

```bash
# 가상환경 설정 (최초 1회)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_mac.txt

# 실행
source .venv/bin/activate
python main.py                  # VLM 30초 간격 (기본)
python main.py --interval 10    # 10초 간격 (M-series Mac 권장)
```

### Jetson Orin Nano Super

```bash
ssh jetson@172.20.10.11

cd ~/edge-vlm-hvac-system
./run.sh                        # venv·DISPLAY·환경변수 자동 감지

# DISPLAY 없으면 자동 headless (창 없이 콘솔+CSV). 강제:
HVAC_HEADLESS=1 ./run.sh

# 상시 가동 (systemd):
sudo cp deploy/hvac.service /etc/systemd/system/
sudo systemctl enable --now hvac
journalctl -u hvac -f
```

### 시작화면 선택지

```
1. 환경 프로파일  →  사무실 / 가정 / 체육시설 / 부대시설
2. 카메라 소스   →  USB / Jetson CSI
3. 동작 모드     →  🎥 실시간 카메라 모드  /  🎬 영상 파일 분석 모드
```

---

## 환경 설정

`.env` 파일을 루트에 생성한다 (`.env.example` 참고):

```bash
cp .env.example .env
```

```env
WEATHER_API_KEY=기상청_API_키    # data.go.kr 초단기실황
```

API 키가 없어도 기본값(외기 20°C)으로 동작한다.

---

## 영상 분석 모드

실제 사무실 영상을 오프라인으로 분석해 **AI 제어 vs 규칙기반(24°C 고정) 비교 리포트**를 자동 생성한다.

```
시작화면 → 영상 파일 분석 모드 → 영상 경로 입력 (Ctrl+V 가능)
→ 초기 실내온도·습도 트랙바 설정 → 분석 시작
```

**분석 방식:**
- 30초 단위 프레임 점프 → VLM 1회 분석
- AI·규칙기반 물리 시뮬레이션 동시 진행 (dt=0.1s × 300스텝 = 30초 물리)
- 완료 후 `results/분석명_날짜시간/` 에 자동 저장

**생성 결과물:**
```
results/분석명_날짜시간/
├── 01_pmv_comparison.png        PMV 시계열 비교
├── 02_indoor_temp.png           실내온도 비교
├── 03_energy_cumulative.png     누적 에너지 소비
├── 04_energy_bar.png            에너지 절감률 막대
├── 05_comfort_rate.png          쾌적 구간 비율
├── 06_activity_distribution.png VLM 감지 활동 분포
├── analysis_log.csv             프레임별 상세 로그
└── summary.txt                  핵심 수치 요약
```

**독립 시나리오 실행 (카메라 없이 JSON 조건으로 테스트):**
```bash
python scenario_runner.py
python scenario_runner.py --scenario scenarios/winter_office.json
python scenario_runner.py --output-dir my_results/
```

---

## Jetson Orin Nano Super 배포

### 환경 정보

| 항목 | 값 |
|------|-----|
| 보드 | Jetson Orin Nano Super 8GB |
| OS | Ubuntu 22.04 (JetPack 6.x) |
| CUDA | 12.6 / Ampere arch 87 |
| GPU | 1024-core, 67 TOPS |
| IP / user | 172.20.10.11 / jetson |

### llama.cpp CUDA 빌드 (최초 1회)

```bash
cd ~/llama.cpp
export PATH=/usr/local/cuda-12.6/bin:$PATH
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6
cmake --build build --config Release -j4
```

### SHT31 온습도 센서

- GPIO Pin3(SDA) / Pin5(SCL), I2C Bus 7, 주소 0x44
- 정밀도: ±0.3°C / ±2% RH
- 센서 연결 시 자동 감지, 없으면 물리 시뮬레이션으로 fallback

### cv2 GTK 백엔드 설정 (최초 1회)

```bash
# PyPI opencv Qt 백엔드 충돌 방지 — 시스템 GTK opencv 링크
ln -sfn /usr/lib/python3/dist-packages/cv2.cpython-310-aarch64-linux-gnu.so \
    ~/edge-vlm-hvac-system/vlm-env/lib/python3.10/site-packages/cv2.so
```

### 필수 환경 변수

```bash
GGML_CUDA_NO_VMM=1    # Jetson 통합 메모리 OOM 방지 (필수)
PYTHONUNBUFFERED=1    # 로그 실시간 출력
```

---

## 실험 결과

### 검증 데이터셋

**Edinburgh Office Monitoring Video Dataset**

> T. Qasim, R. B. Fisher, N. Bhatti, *"Ground-truthing Large Human Behavior Monitoring Datasets,"* Proc. ICPR 2020.  
> http://homepages.inf.ed.ac.uk/rbf/OFFICEDATA/ — License: CC BY-NC-SA

| 항목 | 내용 |
|------|------|
| 해상도 | 1280 × 720, ~1 FPS |
| 총 프레임 | 456,714 (20일치) |
| 공간 | 4개 오피스 위치 |
| 레이블 | bounding box + 행동 (standing/sitting/talking/fallen) |

### 사무실 영상 분석 결과 (78분 / 겨울 조건: 외기 3°C, 초기 실내 10°C)

| 지표 | AI 제어 | 규칙기반 (24°C 고정) |
|------|---------|---------------------|
| 에너지 절감률 | **~29%** | 기준 |
| PMV 쾌적 구간 비율 | **82.4%** | 61.3% |
| 쾌적도 향상 | **+21.1 ppt** | — |
| 평균 실내온도 | 22~24°C 유지 | 24°C 고정 |
| VLM 행동 분류 정확도 | **78.4%** | — |
| 인원 감지 정확도 (YOLO) | **96.3%** | — |

---

## 브랜치 안내

| 브랜치 | 설명 |
|--------|------|
| `main` | 안정 버전 — Mac 개발 환경 기준 |
| `feature/llamacpp-int4-quantization` | Jetson llama.cpp CUDA INT4 (구버전) |
| `feature/final-overhaul` | **현재 활성** — llama-server 상주, 상태머신 제어 연동, 실시간 에너지 모니터, headless 지원 |

---

## 라이선스

MIT License — [LICENSE](LICENSE) 참고
