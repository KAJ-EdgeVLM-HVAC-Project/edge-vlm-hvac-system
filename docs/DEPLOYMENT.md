# 배포 및 운영 가이드 (Deployment & Operations)

> 최종 수정: 2026-06-19  
> 상태: Production Ready

---

## 1. 배포 전략 (Deployment Strategy)

### 1.1 배포 환경 구성

| 환경 | 목적 | 하드웨어 | 배포 브랜치 |
|-----|------|--------|-----------|
| **Development** | 로컬 개발 및 테스트 | 노트북/Linux | feature/* |
| **Staging** | 통합 테스트 | Jetson Nano | develop |
| **Production** | 실제 운영 배포 | Jetson Orin | main (v*.*.*) |

### 1.2 배포 프로세스

```
Feature Branch (개발)
    ↓ (PR 생성)
Code Review & CI/CD 검증
    ↓ (승인)
Merge to Main (릴리스 버전)
    ↓ (Git Tag)
deploy-main.yml 워크플로우 자동 실행
    ├── Pre-deployment Check
    ├── Prepare Deployment Package
    ├── Health Check
    ├── Smoke Test
    └── Deployment Notification
    ↓
Production Ready (배포 완료)
```

---

## 2. 헬스체크 (Health Check)

### 2.1 자동 헬스체크 (CI/CD 파이프라인)

배포 시 자동으로 실행되는 헬스체크 항목:

```python
# Health Check Criteria
1. Python 문법 검증 (Syntax Check)
   - main.py, control_logic.py, vlm_processor.py
   
2. 모듈 임포트 검증
   - 핵심 모듈 정상 로드 여부
   - 외부 의존성 정상 설치 여부
   
3. 환경 프로필 검증
   - PROFILE_CONFIGS 정상 로드
   - 필수 환경 변수 확인
   
4. 스모크 테스트
   - tests/test_core.py 실행
   - 핵심 기능 검증
```

### 2.2 수동 헬스체크 (배포 후)

**Jetson 배포 후 다음을 확인하세요:**

```bash
# 1. 서비스 상태 확인
systemctl status hvac-system
ps aux | grep python

# 2. 로그 확인
tail -f /var/log/hvac-system/main.log

# 3. 센서 연결 확인
python -c "from sensor_interface import SensorInterface; s = SensorInterface(); print(s.read_temperature())"

# 4. API 응답 확인
curl http://localhost:8000/health

# 5. VLM 모델 로드 확인
python -c "from vlm_processor import VLMProcessor; vlm = VLMProcessor(); print('✅ VLM Ready')"
```

### 2.3 헬스체크 응답 코드

| 상태 | HTTP 코드 | 의미 |
|-----|---------|------|
| ✅ Healthy | 200 OK | 시스템 정상 |
| ⚠️ Degraded | 206 Partial | 일부 기능 제한 |
| ❌ Unhealthy | 503 Service Unavailable | 시스템 장애 |

**헬스체크 엔드포인트:**
```
GET /health
Response: {
  "status": "healthy",
  "version": "v1.0.1",
  "components": {
    "vlm": "operational",
    "yolo": "operational",
    "sensors": "operational",
    "controller": "operational"
  },
  "timestamp": "2026-06-19T10:30:00Z"
}
```

---

## 3. 롤백 계획 (Rollback Plan)

### 3.1 롤백 시나리오

| 시나리오 | 증상 | 롤백 방법 | 예상 시간 |
|---------|------|---------|---------|
| **메이저 버전 호환성 문제** | 앱 시작 실패 | Git Tag로 이전 버전 체크아웃 | 5분 |
| **VLM 모델 로드 실패** | 추론 불가 | model/ 디렉토리 복구 | 10분 |
| **센서 드라이버 호환성** | 센서 읽음 오류 | requirements_jetson.txt 이전 버전 설치 | 15분 |
| **제어 로직 오류** | 공조 오작동 | EMPTY 상태로 강제 전환 후 복구 | 2분 |

### 3.2 빠른 롤백 (Quick Rollback)

**방법 1: Git Tag 롤백**
```bash
# 이전 버전 확인
git tag -l | sort -V

# 이전 버전으로 체크아웃
git checkout v1.0.0

# 또는 이전 커밋으로 직접 복귀
git checkout fa83bb0
```

**방법 2: 배포 아티팩트 롤백**
```bash
# GitHub Actions 아티팩트에서 이전 버전 다운로드
# .github/workflows/deploy-main.yml의 "Prepare Deployment Package"에서 생성된 아티팩트 사용

# 배포 매니페스트 확인
cat deployment-manifest.json

# 이전 패키지 복원
unzip deployment-package-v1.0.0.zip -d /opt/hvac-system/
systemctl restart hvac-system
```

**방법 3: 상태 머신 강제 리셋**
```python
# 긴급 상황: 공조 시스템을 안전 상태로 즉시 전환
python -c "
from state_machine import StateMachine
from control_logic import ControlLogic

sm = StateMachine()
sm.force_state('EMPTY')  # 모든 센서 무시, 대기 상태로 강제 전환

ctrl = ControlLogic()
ctrl.fan_speed = 0  # 팬 정지
ctrl.target_temperature = None  # 제어 해제
print('✅ System reset to EMPTY state')
"
```

### 3.3 롤백 검증

```bash
# 롤백 후 반드시 확인할 사항:
1. ✅ 시스템 시작 성공
2. ✅ 로그에 오류 없음
3. ✅ 센서 데이터 수신 정상
4. ✅ 제어 명령 정상
5. ✅ 대시보드 응답 정상
```

---

## 4. 배포 체크리스트

### 배포 전

- [ ] 모든 PR 리뷰 완료
- [ ] CI/CD 파이프라인 통과
- [ ] 스모크 테스트 성공
- [ ] 버전 번호 업데이트 (CHANGELOG.md)
- [ ] Git Tag 생성 (v*.*.*)

### 배포 중

- [ ] deploy-main.yml 워크플로우 실행
- [ ] Pre-deployment Check 통과
- [ ] Health Check 통과
- [ ] 배포 로그 확인

### 배포 후

- [ ] 시스템 정상 시작 확인
- [ ] 센서 데이터 수신 확인
- [ ] 대시보드 접근 가능 확인
- [ ] 에러 로그 확인
- [ ] 팀에 배포 알림

---

## 5. 배포 로깅 및 모니터링

### 배포 로그 위치

```
Local Development:
  logs/deployment.log

Jetson Production:
  /var/log/hvac-system/deployment.log
  /var/log/hvac-system/main.log
  /var/log/hvac-system/error.log
```

### 배포 성공 신호

```log
[2026-06-19 10:30:00] ✅ Pre-deployment validation passed
[2026-06-19 10:30:05] ✅ Deployment package prepared
[2026-06-19 10:30:10] ✅ Health check completed
[2026-06-19 10:30:15] ✅ Smoke tests passed
[2026-06-19 10:30:20] ✅ Main branch deployment successful (v1.0.1)
```

---

## 6. 긴급 연락 및 지원

| 상황 | 담당자 | 연락 방법 |
|-----|------|---------|
| 배포 실패 | 김철호 (PM) | GitHub Issues |
| 센서 오류 | 김민서 (열환경) | Slack @eng-hvac |
| AI 추론 오류 | 김준경 (AI) | GitHub Issues |
| 긴급 롤백 | 김철호 (PM) | 즉시 GitHub Actions 취소 후 수동 롤백 |

---

## 7. 배포 이력 (Deployment History)

| 버전 | 날짜 | 상태 | 배포자 | 주요 변경 |
|-----|------|------|------|---------|
| v1.0.1 | 2026-06-19 | ✅ Success | Kim CH | ADR 추가, 배포 파이프라인 구축 |
| v1.0.0 | 2026-06-15 | ✅ Success | Kim JG | 초기 릴리스 |

---

**Last Updated**: 2026-06-19  
**Next Review**: 2026-07-19
