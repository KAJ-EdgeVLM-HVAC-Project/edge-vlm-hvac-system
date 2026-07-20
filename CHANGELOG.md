# 변경 이력 (Changelog)

이 프로젝트의 주요 변경 사항을 기록합니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를,
버전은 [Semantic Versioning](https://semver.org/lang/ko/)을 따릅니다.

## [Unreleased]

### Fixed (버그 수정)
- **수증기 분압 단위 오류** (`thermal_engine.py`): ISO 7730 Annex D의 `pa`는 Pa 단위인데
  `rh/100`으로 계산해 100배 작았음. 증발 열손실이 과대평가되어 **습도가 PMV에 거의
  반영되지 않던 상태**를 수정.
- **난방 하한 목표온도 고정** (`control_logic.py`): 난방 하한 분기가 목표온도를 24°C로
  하드코딩해 PMV에 따른 동적 목표온도가 무시되던 문제 수정.

### Changed (변경)
- **대사율 기준 정정** (`vlm_processor.py`): `sitting` 1.0 → **1.2**
  (ISO 7730 Table B.1 *sedentary office work*). 기존 1.0은 *seated, relaxed*(휴식)
  수준이라 사무 환경과 맞지 않았고, 이로 인해 냉방 시작이 27.3°C까지 밀렸음
  (수정 후 26.0°C). `standing` 1.2 → 1.4, `walking` 1.5 → 1.7도 함께 정정.
- **인원 감지 GPU 이관**: YOLOv8n(CPU, 1,265ms) → **YOLO26s TensorRT 엔진**(GPU, 약 39ms).
  보드 PyTorch가 Orin(sm_87)용 빌드가 아니어서 GPU 실행이 불가했던 것을
  torch 없이 TensorRT를 직접 호출하는 방식으로 우회. 감지 주기 3초 → **0.1초**.
- **룰베이스 기준 재정의** (`energy_monitor.py`): 연중 24°C 고정 → 계절별
  (냉방 24°C / 난방 25°C, 외기 20°C 기준) + 공실 정지.
- **영상 분석 모드 실시간화** (`main.py`): 실시간 카메라 모드와 동일한 파이프라인으로
  재작성. Jetson 하드웨어 디코더 사용, 벽시계 기준 프레임 드롭, dt 기반 물리 계산.

### Added (추가)
- **`occupancy_tracker.py`** — 재실 인원 추적 모듈.
  이동 예측(등속도 + EMA 평활)으로 프레임 간 동일인을 판정하고, 겹침(IoU)·거리
  2단계 비용행렬에 **헝가리안 최적 배정**을 적용해 전체 비용 합이 최소인 조합을
  선택한다. IoU 단독 매칭에서 빠른 이동 시 1명이 2명으로 계수되던 문제 해결.
  0↔1 인원 전환 지점을 학습해 **출입구를 자동 추정**(사전 매핑 불필요).
- **기류 연동** (`thermal_engine.py`): 팬 단계 → 체감 기류속도(0.10~0.25 m/s)를
  PMV에 반영. 냉방 중에도 정지공기로 계산하던 모순 제거.
- **외기 적응 보정** (`thermal_engine.py`): 외기온에 따라 목표 PMV 기준선을 ±0.4 이동.
- **다중 재실자 대응** (`main.py`): 인원별 VLM 라운드로빈 분석 후 개인별 PMV의
  중앙값을 채택.

### Changed (기존 기록)
- 에너지 전력 모델을 **압축기 부하 기반**으로 개선: `P = P_fan + P_comp,rated × load`.
  기존 팬 단계 전용(800/1200/1600 W) 모델은 압축기 부하와 설정온도를 반영하지
  못했음. 부하율은 실내로 새어드는 열÷냉난방 능력으로 산정해 설정온도·외기조건이
  전력에 반영됨. 팬 40/70/110 W, 압축기 정격 1200 W.
- 에든버러 영상 재분석 결과 갱신: 에너지 절감 **47.7% → 68.2%**
  (AI 254.3 Wh vs 룰베이스 799.9 Wh). 쾌적율(85.9%/84.6%)·실내온도는 불변.

## [1.0.0] - 2026-06-19

현장미러형 프로젝트 최종 릴리스. Jetson Orin Nano Super 실기기 배포·검증 완료.

### Added (추가)
- Qwen3-VL-2B INT4 GGUF 지원 및 자동 모델 탐색(`qwen3` 우선 선택)
- llama-server 상주 추론(모델 1회 로드 + HTTP, json_schema 강제)
- 옷차림 5단계 분류(민소매~아우터) → clo 0.35~1.3 매핑
- 상태 머신(ARRIVAL/PRE_DEPARTURE/LUNCH_BREAK)의 제어 로직 반영
- 영상 분석 모드: AI vs 룰베이스 에너지 비교 리포트 자동 생성
- headless 모드(디스플레이 없으면 콘솔/CSV만, systemd 상시가동)
- 라이트 테마 대시보드 UI 전면 재설계

### Changed (변경)
- 보드 입력 해상도 상향: 카메라 720p, YOLO imgsz 768, VLM 입력 1024폭
- 달력 기반 계절 CLO → 외부온도 기반 fallback으로 단순화
- 카메라(라이브) 모드에서 에너지 절감률 표시 제거(영상 분석 전용 지표)

### Fixed (수정)
- MPS OOM으로 VLM 스레드가 죽어 clo가 기본값에 고정되던 문제
- MPS VLM 출력 JSON 깨짐(프롬프트 echo 제거)
- Jetson USB 카메라 미지원 + 8MP 캡처 부하(폴백·리사이즈 추가)
- 인원 0명 표시 버그(YOLO 감지 즉시 화면 반영)
- 실내 25°C에서 난방이 켜지는 모순(난방 진입 상한 추가)

### Removed (제거)
- 공기질(PM10/PM2.5) API 연동, 창문 개폐 권장 기능
- TensorRT-LLM 백엔드(Orin Nano 비실용 → llama.cpp INT4로 대체)

## [0.9.0] - 2026-05-14

- Jetson Orin Nano Super 최초 배포(llama.cpp CUDA INT4)
- 영상 파일 분석 모드 + 논문용 리포트 자동 생성
- PMV 기반 동적 목표온도 제어 + PID 팬 제어

## [0.5.0] - 2026-03

- Mac(Apple Silicon MPS)에서 VLM 추론 프로토타입
- ISO 7730 PMV 엔진, HVAC 물리 시뮬레이터, YOLO 인원 감지

[1.0.0]: https://github.com/KAJ-EdgeVLM-HVAC-Project/edge-vlm-hvac-system/releases/tag/v1.0.0
