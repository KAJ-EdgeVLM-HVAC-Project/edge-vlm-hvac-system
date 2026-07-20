# 운영 Runbook

Edge VLM HVAC 시스템의 배포·헬스체크·롤백·관측성 운영 문서.
대상 환경: **NVIDIA Jetson Orin Nano Super** (JetPack 6.x, user `jetson`).

---

## 1. 배포 (Deployment)

본 시스템은 웹 서버가 아닌 **엣지 디바이스 상주 애플리케이션**으로, 보드에서
직접 구동된다. `main` 브랜치(또는 릴리스 태그)를 보드에서 pull 하여 배포한다.

```bash
# 보드 접속
ssh jetson@<board-ip>

cd ~/edge-vlm-hvac-system
git fetch origin
git checkout main && git pull          # 또는: git checkout v1.0.0
./run.sh                                # venv/DISPLAY 자동 감지, headless 자동 전환
```

**상시 가동(무인 운영)** 은 systemd 서비스로 등록한다(`HVAC_HEADLESS=1`):

```bash
sudo systemctl enable hvac.service
sudo systemctl start  hvac.service
```

최초 1회 필요: llama.cpp 빌드 + Qwen3-VL-2B GGUF를
`~/llama.cpp/models/Qwen3-VL-2B/` 에 배치 (자동 탐색됨). 자세한 내용은 [README](../README.md) 참고.

---

## 2. 헬스체크 (Health Check)

| 점검 항목 | 정상 기준 | 확인 방법 |
|---|---|---|
| VLM 추론 서버 | HTTP 200 | `curl http://127.0.0.1:8090/health` |
| 추론 동작 | clean JSON 출력 | 콘솔 로그 `[VLM OUTPUT] {"clothing":...}` |
| 데이터 기록 | CSV 행 증가 | `tail -f hvac_system_performance.csv` |
| 프로세스 생존 | active (running) | `systemctl status hvac.service` |
| VLM 추론 지연 | 약 1.2초/회 (Jetson, Qwen3-VL-2B INT4) | 콘솔 `[N] x.xs` 출력 |
| YOLO 추론 지연 | 약 39ms/회 (Jetson, TensorRT GPU) | 0.1초 주기 실행 |

llama-server 자체 헬스 엔드포인트(`/health`)가 200을 반환하지 않으면 추론
불가 상태이며, 본 시스템은 자동으로 CLI 추론 모드로 폴백한다.

---

## 3. 롤백 계획 (Rollback)

**롤백 트리거 기준** (아래 중 하나라도 충족 시 롤백):
- 배포 후 시스템이 부팅/추론에 반복 실패 (헬스체크 연속 3회 실패)
- VLM 추론이 OOM 등으로 지속 크래시
- 제어 로직 회귀로 냉난방이 비정상 동작

**롤백 절차:**
```bash
cd ~/edge-vlm-hvac-system
sudo systemctl stop hvac.service

# 직전 안정 릴리스로 복귀
git checkout v1.0.0          # 또는 직전 정상 커밋 해시
sudo systemctl start hvac.service

# 헬스체크로 정상 복귀 확인
curl http://127.0.0.1:8090/health
```

**책임자:** 배포를 수행한 팀원이 롤백까지 책임진다. 모델 파일(GGUF)은 코드와
분리되어 있어 롤백 시 재다운로드가 필요 없다.

**원격 롤백(원격 저장소):** 잘못된 머지가 `main`에 들어간 경우
`git revert <merge-commit>` 로 되돌리는 PR을 생성한다. force-push는 금지.

---

## 4. 관측성 (Observability)

| 종류 | 위치 | 내용 |
|---|---|---|
| **메트릭/로그** | `hvac_system_performance.csv` | 타임스탬프·상태·인원·PMV·온습도·제어·옷차림(clo)·활동(met) 등 프레임 단위 기록 |
| **추론 로그** | `/tmp/llama-server-hvac.log` | llama-server 모델 로드·추론 타이밍·토큰 속도 |
| **실시간 대시보드** | OpenCV 창 2개 | 운영자 대시보드(센서·제어·VLM 컨텍스트) + 사용자 UI. headless 시 콘솔 출력 |
| **분석 리포트** | `results/<name>/` | 영상 분석 모드의 PMV·에너지·쾌적율·활동분포 그래프 + summary.txt |

CSV 스키마가 변경되면 기존 파일은 자동 백업 후 재생성된다(`initialize_csv`).

---

## 5. 자주 발생하는 문제 (Troubleshooting)

| 증상 | 원인 | 조치 |
|---|---|---|
| 추론 시작 시 OOM | 통합 메모리 CUDA VMM | `GGML_CUDA_NO_VMM=1` (run.sh가 설정) |
| 카메라 안 열림 | CSI/USB 혼용 | main.py가 CSI→USB 자동 폴백 |
| 창 안 뜸 | DISPLAY 없음 | headless 모드로 자동 전환(콘솔+CSV) |
| 추론 느림(30초+) | llama-server 미기동 | `/tmp/llama-server-hvac.log` 확인, 바이너리 존재 점검 |
