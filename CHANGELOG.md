# 변경 이력 (Changelog)

이 프로젝트의 주요 변경 사항을 기록합니다.
형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를,
버전은 [Semantic Versioning](https://semver.org/lang/ko/)을 따릅니다.

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
