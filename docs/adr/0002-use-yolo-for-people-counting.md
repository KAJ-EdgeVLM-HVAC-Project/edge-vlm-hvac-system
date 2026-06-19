# ADR-0002: YOLOv8n 기반 인원 감지 도입

## Status

Accepted

---

## Context

초기 설계에서는 VLM(Qwen2-VL-2B)이 인원 수(people count)까지 추출하도록 프롬프트를 구성했다.
그러나 실제 테스트 결과 다음 문제가 확인되었다:

* VLM 인원 감지 정확도 불안정 (~60~70%) — 할루시네이션으로 엉뚱한 숫자 반환
* VLM 추론 주기가 30~60초/회 → 사람이 들어오고 나가는 이벤트를 실시간으로 반영 불가
* 인원 수는 StateManager 상태 전이(EMPTY/ARRIVAL)의 핵심 입력값 → 오류가 제어 오동작으로 직결

---

## Decision

인원 수 감지는 **YOLOv8n(ultralytics)** 이 전담한다.

* 입력: 카메라 프레임 (BGR numpy array)
* 감지 대상: class 0 (person)
* 실행 주기: 매 5프레임 (메인 루프 내 동기 실행)
* 설정: imgsz=320 conf=0.35 (CPU 기준), imgsz=640 (Jetson TRT)
* 반환: 감지된 인원 수 (int), YOLO 미사용 시 -1 반환 → 이전 값 유지(graceful fallback)

VLM 프롬프트에서 `people` 필드를 제거하여 추론 부담을 낮추고 응답 안정성을 높인다.

---

## Consequences

### 장점

* 인원 감지 정확도 95%+
* CPU에서 10~20fps, Jetson TRT에서 30fps+ 실시간 처리
* 할루시네이션 없음 (전용 객체 탐지 모델)
* YOLO 미설치 시 자동 폴백 → 기존 값 유지로 시스템 중단 없음
* VLM 프롬프트 5개 필드로 단순화 → 응답 파싱 안정성 향상

### 단점

* 의존성 추가: `ultralytics` (~6MB 모델 최초 다운로드 필요)
* 군중 밀집 환경(겹침)에서 과소 감지 가능 — 본 프로젝트 사무실 환경에서는 허용 범위

---

## 구현 위치

* `yolo_detector.py` — YOLODetector 클래스
* `convert_tensorrt.py` — Jetson TRT FP16 엔진 변환 (`--yolo` 옵션)
* `main.py` — `YOLO_EVERY_N_FRAMES = 5` 주기 실행, `count_source` CSV 기록
