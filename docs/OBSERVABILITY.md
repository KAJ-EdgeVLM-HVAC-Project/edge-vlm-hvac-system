# 관측성 (Observability) 가이드

> 최종 수정: 2026-06-19  
> 상태: Production Ready

---

## 1. 로깅 (Logging)

### 1.1 로그 레벨 및 포맷

```python
# logging 설정 예제 (main.py에 추가)
import logging
import sys
from datetime import datetime

# 로그 포맷
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 기본 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler('/var/log/hvac-system/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
```

### 1.2 로그 위치 및 파일 구조

**Local Development:**
```
logs/
├── main.log                    # 애플리케이션 메인 로그
├── vlm_processor.log          # VLM 추론 로그
├── control_logic.log          # 제어 로직 로그
├── error.log                  # 에러 로그
└── performance.log            # 성능 메트릭 로그
```

**Jetson Production:**
```
/var/log/hvac-system/
├── main.log                    # 애플리케이션 메인 로그
├── vlm_processor.log          # VLM 추론 로그
├── control_logic.log          # 제어 로직 로그
├── error.log                  # 에러 로그
├── performance.log            # 성능 메트릭 로그
├── deployment.log             # 배포 로그
└── archive/                   # 로그 아카이브 (7일 단위)
```

### 1.3 주요 로깅 포인트

```python
# VLM 추론 로깅
logger.info(f"VLM inference started - image_size: {img.shape}")
logger.info(f"VLM inference completed - clo: {clo}, met: {met}, time: {elapsed}ms")
logger.error(f"VLM inference failed: {error}")

# 제어 로직 로깅
logger.info(f"State transition: {prev_state} -> {new_state}")
logger.info(f"PID control output - fan_speed: {fan}, temp: {target_temp}°C")
logger.warning(f"Temperature threshold exceeded: {current_temp}°C > {max_temp}°C")

# 센서 데이터 로깅
logger.debug(f"Sensor reading - temp: {temp}°C, humidity: {humidity}%")

# 에러 처리
try:
    # 작업
except Exception as e:
    logger.error(f"Operation failed: {str(e)}", exc_info=True)
```

### 1.4 로그 로테이션

**로그 로테이션 설정 (systemd 환경):**
```bash
# /etc/logrotate.d/hvac-system
/var/log/hvac-system/*.log {
    daily                    # 매일 로테이션
    rotate 7                 # 7일치 보관
    compress                 # 압축
    missingok               # 파일 없어도 무시
    notifempty              # 비어있으면 로테이션 안함
    create 0640 hvac hvac   # 새 파일 권한
    postrotate
        systemctl reload hvac-system > /dev/null 2>&1 || true
    endscript
}
```

---

## 2. 메트릭 (Metrics)

### 2.1 주요 성능 메트릭

| 메트릭 | 수집 주기 | 임계값 | 알림 |
|-------|---------|------|------|
| **VLM 추론 시간** | 30초마다 | < 5s | > 10s |
| **센서 응답 시간** | 10초마다 | < 100ms | > 500ms |
| **제어 루프 주기** | 실시간 | 100-200ms | > 500ms |
| **메모리 사용량** | 1분마다 | < 1.5GB | > 2GB |
| **GPU 사용률 (Jetson)** | 1분마다 | < 80% | > 95% |
| **CPU 온도** | 1분마다 | < 70°C | > 85°C |

### 2.2 메트릭 수집 구현

```python
import time
import psutil
from collections import deque

class MetricsCollector:
    def __init__(self):
        self.vlm_times = deque(maxlen=100)  # 최근 100개
        self.sensor_times = deque(maxlen=100)
        self.control_loop_times = deque(maxlen=100)
        
    def record_vlm_inference(self, elapsed_ms):
        self.vlm_times.append(elapsed_ms)
        avg = sum(self.vlm_times) / len(self.vlm_times)
        if elapsed_ms > 5000:  # 5초 초과
            logger.warning(f"VLM inference slow: {elapsed_ms}ms (avg: {avg}ms)")
    
    def get_system_metrics(self):
        return {
            "timestamp": time.time(),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_temp": self.get_cpu_temp(),
            "vlm_avg_ms": sum(self.vlm_times) / len(self.vlm_times) if self.vlm_times else 0,
        }
    
    def get_cpu_temp(self):
        try:
            import subprocess
            result = subprocess.check_output(["cat", "/sys/class/thermal/thermal_zone0/temp"])
            return int(result) / 1000  # 밀리도 -> 도
        except:
            return None

metrics = MetricsCollector()
```

### 2.3 메트릭 내보내기 (Prometheus 형식)

```python
# /metrics 엔드포인트 구현
from flask import Flask, Response

app = Flask(__name__)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus 형식의 메트릭 제공"""
    metrics_data = metrics.get_system_metrics()
    
    output = f"""# HELP hvac_vlm_inference_ms VLM 추론 시간 (밀리초)
# TYPE hvac_vlm_inference_ms gauge
hvac_vlm_inference_ms {metrics_data['vlm_avg_ms']}

# HELP hvac_memory_percent 메모리 사용률 (%)
# TYPE hvac_memory_percent gauge
hvac_memory_percent {metrics_data['memory_percent']}

# HELP hvac_cpu_percent CPU 사용률 (%)
# TYPE hvac_cpu_percent gauge
hvac_cpu_percent {metrics_data['cpu_percent']}

# HELP hvac_cpu_temp CPU 온도 (°C)
# TYPE hvac_cpu_temp gauge
hvac_cpu_temp {metrics_data['cpu_temp'] or 0}
"""
    return Response(output, mimetype='text/plain')
```

---

## 3. 대시보드 (Dashboard)

### 3.1 실시간 대시보드 (웹 기반)

**dashboard.py**에 구현된 기능:

```
┌─────────────────────────────────────────┐
│        HVAC 실시간 대시보드              │
├─────────────────────────────────────────┤
│                                         │
│  📊 시스템 상태                          │
│  ├─ 현재 상태: STEADY                  │
│  ├─ 운영 환경: Office                  │
│  └─ 시스템 온도: 22.5°C                │
│                                         │
│  🌡️  센서 데이터                        │
│  ├─ 실내 온도: 22.5°C                  │
│  ├─ 실내 습도: 45%                     │
│  └─ CO2: 450ppm                        │
│                                         │
│  👥 사람 감지                            │
│  ├─ 인원 수: 3명                        │
│  ├─ 활동 강도: Medium                  │
│  └─ 착의량(CLO): 1.2                   │
│                                         │
│  🎯 제어 정보                            │
│  ├─ 팬 속도: 65%                       │
│  ├─ 목표 온도: 23°C                    │
│  └─ PMV 지수: -0.2 (쾌적)              │
│                                         │
└─────────────────────────────────────────┘
```

### 3.2 대시보드 구성

**주요 화면:**

1. **시스템 상태 (System Status)**
   - 현재 상태 머신 상태
   - 운영 환경 프로필
   - 시스템 온도/습도

2. **실시간 제어 (Real-time Control)**
   - 팬 속도 조절
   - 목표 온도 설정
   - PMV 편의 조정 (±0.5)

3. **성능 그래프 (Performance Graphs)**
   - 온도 변화 추이
   - 에너지 사용량
   - 사람 활동 패턴

4. **로그 뷰 (Log Viewer)**
   - 최근 100개 이벤트
   - 에러 메시지
   - 상태 전환 기록

### 3.3 대시보드 접근

```
Local Development:
  http://localhost:5000 (operator view)
  http://localhost:5001 (user remote control)

Production (Jetson):
  http://<jetson-ip>:5000 (operator view)
  http://<jetson-ip>:5001 (user remote control)
```

### 3.4 대시보드 데이터 API

```python
# GET /api/status - 시스템 상태
{
  "timestamp": "2026-06-19T10:30:00Z",
  "state": "STEADY",
  "environment": "office",
  "temperature": 22.5,
  "humidity": 45,
  "fan_speed": 65,
  "target_temperature": 23.0,
  "pmv_index": -0.2,
  "comfort_level": "comfortable",
  "people_count": 3,
  "activity_level": "medium"
}

# GET /api/metrics - 성능 메트릭
{
  "timestamp": "2026-06-19T10:30:00Z",
  "vlm_inference_ms": 2500,
  "memory_percent": 62.3,
  "cpu_percent": 45.2,
  "cpu_temp": 68.5
}

# GET /api/logs?limit=100 - 로그 조회
{
  "logs": [
    {
      "timestamp": "2026-06-19T10:30:00Z",
      "level": "INFO",
      "message": "State transition: ARRIVAL -> STEADY"
    }
  ]
}
```

---

## 4. 모니터링 및 알림

### 4.1 알림 임계값

```yaml
Alerts:
  - name: HighVLMLatency
    condition: vlm_inference_ms > 5000
    severity: WARNING
    action: "Performance degradation alert"
  
  - name: HighMemoryUsage
    condition: memory_percent > 85
    severity: WARNING
    action: "Consider system restart"
  
  - name: SystemOverheat
    condition: cpu_temp > 85
    severity: CRITICAL
    action: "Reduce processing load"
  
  - name: SensorFailure
    condition: sensor_read_error_count > 5
    severity: CRITICAL
    action: "Manual intervention required"
```

### 4.2 알림 전송

```python
def send_alert(alert_type, message, severity):
    logger.log(
        level=logging.WARNING if severity == "WARNING" else logging.CRITICAL,
        msg=f"[{alert_type}] {message}"
    )
    
    if severity == "CRITICAL":
        # 긴급 알림: 대시보드 팝업, 로그 기록
        notify_critical(message)
```

---

## 5. 관측성 체크리스트

### 배포 전

- [ ] 로깅 설정 확인
- [ ] 메트릭 수집 활성화
- [ ] 대시보드 접근성 테스트
- [ ] 로그 로테이션 설정

### 배포 후

- [ ] 로그 파일 생성 확인
- [ ] 메트릭 엔드포인트 정상 작동
- [ ] 대시보드 데이터 수신 확인
- [ ] 알림 기능 테스트

---

**Last Updated**: 2026-06-19
