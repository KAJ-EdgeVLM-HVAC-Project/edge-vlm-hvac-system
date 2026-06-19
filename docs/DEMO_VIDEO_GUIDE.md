# 영상 데모 준비 가이드

> 최종 수정: 2026-06-19  
> 목표: 3분 이내 데모 영상 제작 및 업로드

---

## 1. 데모 영상 요구사항

### 1.1 영상 사양

| 항목 | 요구사항 | 비고 |
|-----|--------|------|
| **지속시간** | 3분 이내 (180초) | 최대 범위 |
| **해상도** | 1080p (1920×1080) 이상 | 가능하면 4K |
| **프레임율** | 30fps 이상 | 부드러운 재생 |
| **코덱** | H.264 또는 VP9 | 웹 호환성 |
| **포맷** | MP4 또는 WebM | GitHub 호환 |
| **파일 크기** | < 300MB | 업로드 용이 |
| **음성** | 선택사항 (영어/한국어) | 자막 권장 |

### 1.2 데모 콘텐츠 구성

**추천 시나리오 (총 180초):**

```
00:00 - 00:15  (15초) | 오프닝
  ├─ 프로젝트 타이틀 표시
  ├─ "엣지 VLM 및 IoT 융합 기반 지능형 공조 제어 시스템"
  └─ 팀원 정보 표시

00:15 - 00:45  (30초) | 시스템 구성 및 아키텍처
  ├─ 하드웨어: Jetson Orin
  ├─ VLM (Qwen2-VL-2B) + YOLO (YOLOv8n) 다이어그램
  ├─ 실시간 추론 파이프라인 설명
  └─ 5단계 상태 머신 다이어그램

00:45 - 02:15  (90초) | 실제 운영 데모 (핵심)
  ├─ 1단계: 시스템 시작 및 대시보드 화면 (15초)
  │   ├─ Jetson 터미널에서 python main.py 실행
  │   ├─ 초기 상태: EMPTY
  │   └─ 대시보드 로딩
  │
  ├─ 2단계: 실시간 센서 데이터 표시 (20초)
  │   ├─ 온습도 센서 데이터 수신
  │   ├─ 외부 날씨 API 데이터
  │   └─ 메트릭 대시보드
  │
  ├─ 3단계: 인원 감지 및 활동 인식 (20초)
  │   ├─ 카메라 영상 피드 (보안 처리)
  │   ├─ YOLOv8n: 인원 수 감지 (예: 3명 감지)
  │   ├─ VLM: 활동 분류 (예: 운동 중)
  │   └─ 대시보드 업데이트
  │
  ├─ 4단계: 자동 공조 제어 (20초)
  │   ├─ PMV 지수 계산 (열 쾌적성)
  │   ├─ PID 제어기 작동
  │   ├─ 팬 속도 자동 조절
  │   ├─ 목표 온도 자동 설정
  │   └─ 에너지 사용량 감소 그래프
  │
  ├─ 5단계: 상태 머신 전환 시뮬레이션 (15초)
  │   ├─ EMPTY → ARRIVAL (인원 진입)
  │   ├─ ARRIVAL → STEADY (안정화)
  │   ├─ STEADY → LUNCH_BREAK (점심시간)
  │   └─ 각 상태에서 자동 제어 변화
  │
  └─ 6단계: 사용자 인터페이스 데모 (10초)
      ├─ "더워요" 버튼 클릭 → PMV 조정
      ├─ "추워요" 버튼 클릭 → PMV 조정
      └─ 대시보드 반영 (실시간)

02:15 - 02:45  (30초) | 시나리오 분석 결과
  ├─ 겨울철 사무실 시나리오: 에너지 절감 30%
  ├─ 여름철 사무실 시나리오: 쾌적도 향상
  ├─ 헬스장 시나리오: 활동감지 정확도 95%
  └─ 주방 시나리오: 열원 감지 및 환기 자동화

02:45 - 03:00  (15초) | 클로징
  ├─ 주요 성과 요약
  ├─ 기술 스택 재확인
  └─ "Thank You / 감사합니다" 표시
```

---

## 2. 영상 제작 방법

### 2.1 도구 선택

**Option A: OBS Studio (무료, 권장)**
```bash
# Ubuntu/Jetson
sudo apt install obs-studio

# macOS
brew install obs

# Windows
choco install obs-studio
```

**Option B: FFmpeg (명령줄)**
```bash
# 스크린 + 터미널 녹화
ffmpeg -f gdigrab -framerate 30 -i desktop \
  -f dshow -i "마이크" \
  -c:v libx264 -crf 23 \
  -c:a aac output.mp4
```

**Option C: 스마트폰 + 편집**
- iPhone: iMovie, CapCut
- Android: CapCut, Adobe Premiere Rush

### 2.2 제작 단계별 가이드

#### Step 1: 콘텐츠 준비
```bash
# 데모 환경 설정
cd /path/to/edge-vlm-hvac-system

# 시뮬레이션 모드 활성화
python -c "
from sensor_interface import SensorInterface
si = SensorInterface()
si.MODE = 'simulate'  # 실제 센서 대신 시뮬레이션 데이터
print('✅ Demo environment ready')
"
```

#### Step 2: 녹화 설정 (OBS)
```
1. Settings > Output
   - Video Bitrate: 8000-10000 kbps
   - Encoder: NVIDIA NVENC (Jetson) / H.264
   
2. Settings > Video
   - Resolution: 1920x1080
   - FPS: 30

3. Scene Setup:
   - Source 1: 디스플레이 캡처 (대시보드)
   - Source 2: 윈도우 캡처 (터미널)
```

#### Step 3: 녹화 실행
```bash
# 1. 터미널에서 시스템 시작
python main.py

# 2. OBS에서 녹화 시작
# 3. 데모 시나리오 진행
# 4. 3분 경과 후 종료
```

#### Step 4: 후반 작업
```bash
# 비디오 편집 (ffmpeg)
ffmpeg -i demo_raw.mp4 \
  -c:v libx264 -crf 23 \
  -c:a aac -b:a 128k \
  -s 1920x1080 \
  demo_final.mp4

# 파일 크기 확인
ls -lh demo_final.mp4
```

### 2.3 편집 팁

```
자막 추가 (ffmpeg):
  ffmpeg -i demo_final.mp4 \
    -vf "subtitles=demo.srt" \
    demo_with_subtitles.mp4

음성 추가:
  ffmpeg -i demo_final.mp4 \
    -i narration.mp3 \
    -c:v copy -c:a aac \
    demo_with_audio.mp4

로고/워터마크 추가:
  ffmpeg -i demo_final.mp4 \
    -i logo.png \
    -filter_complex "overlay=10:10" \
    demo_branded.mp4
```

---

## 3. 업로드 및 공유

### 3.1 GitHub Release에 업로드

```bash
# 1. 릴리스 태그 생성
git tag -a v1.0.1-demo -m "Add demo video for v1.0.1"
git push origin v1.0.1-demo

# 2. GitHub CLI로 업로드
gh release create v1.0.1-demo \
  --title "HVAC System Demo v1.0.1" \
  --notes "실시간 VLM 및 IoT 기반 공조 제어 시스템 데모" \
  demo_final.mp4

# 또는 웹에서 직접 업로드
# https://github.com/KAJ-EdgeVLM-HVAC-Project/edge-vlm-hvac-system/releases
```

### 3.2 docs/demo 폴더에 업로드

```bash
# 파일 구조
docs/demo/
├── vlm_hvac_demo_v1.0.1.mp4      # 메인 데모
├── demo_subtitles_ko.srt         # 한국어 자막
├── demo_subtitles_en.srt         # 영어 자막
└── README.md                     # 데모 설명

# 업로드
git add docs/demo/
git commit -m "docs: Add HVAC system demo video (v1.0.1)"
git push origin main
```

### 3.3 README에 데모 링크 추가

```markdown
## 📹 실시간 데모

[![Watch the video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

또는 로컬 데모:
- **파일**: [vlm_hvac_demo.mp4](docs/demo/vlm_hvac_demo_v1.0.1.mp4)
- **GitHub Release**: [Demo v1.0.1](https://github.com/.../releases/tag/v1.0.1-demo)
```

---

## 4. 영상 체크리스트

### 제작 전

- [ ] 시스템 정상 작동 확인
- [ ] 대시보드 UI 최종 점검
- [ ] 시뮬레이션 데이터 준비
- [ ] 녹화 환경 세팅 완료

### 제작 중

- [ ] 시작과 끝이 명확
- [ ] 모든 주요 기능 포함
- [ ] 텍스트 및 자막 명확
- [ ] 오디오 품질 확인 (있을 경우)

### 제작 후

- [ ] 파일 크기 < 300MB
- [ ] 재생 품질 확인
- [ ] 오타/오류 없음
- [ ] 3분 이내 (180초)

### 업로드

- [ ] GitHub Release에 업로드
- [ ] README에 링크 추가
- [ ] docs/demo 폴더에 저장
- [ ] 메타데이터 추가 (제목, 설명, 태그)

---

## 5. 데모 영상 배포 체크리스트

```yaml
Demo Video Checklist:
  Format:
    - Resolution: 1920x1080 ✅/❌
    - Duration: ≤ 180 seconds ✅/❌
    - File size: ≤ 300MB ✅/❌
    - Codec: H.264/VP9 ✅/❌

  Content:
    - System Architecture shown ✅/❌
    - Real-time inference demo ✅/❌
    - Sensor data visualization ✅/❌
    - Control logic in action ✅/❌
    - UI demo (user buttons) ✅/❌
    - Scenario results ✅/❌

  Upload:
    - GitHub Release ✅/❌
    - docs/demo folder ✅/❌
    - README link ✅/❌
    - Metadata added ✅/❌

  Quality:
    - No audio artifacts ✅/❌
    - Subtitles clear ✅/❌
    - No copyright issues ✅/❌
    - Playback verified ✅/❌
```

---

**Last Updated**: 2026-06-19

### 다음 단계
1. 위 가이드에 따라 영상 제작
2. docs/demo/vlm_hvac_demo.mp4 생성
3. GitHub Release에 업로드
4. README에 링크 추가
5. PR로 최종 병합
