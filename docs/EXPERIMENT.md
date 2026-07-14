# 실험: 실제 사무실 영상 기반 검증

VLM 기반 AI 제어가 고정 설정온도 룰베이스 대비 에너지·쾌적도에서 어떤 차이를
보이는지, **실제 사무실 CCTV 영상**으로 영상 분석 모드를 통해 검증했다.

---

## 1. 데이터셋 출처 (Attribution)

- **이름:** Edinburgh Office Monitoring Video Dataset (OFFICEDATA)
- **인용:** T. Qasim, R. B. Fisher, N. Bhatti, *"Ground-truthing Large Human
  Behavior Monitoring Datasets,"* Proc. ICPR 2020.
- **제공:** School of Informatics, University of Edinburgh
- **URL:** http://homepages.inf.ed.ac.uk/rbf/OFFICEDATA/
- **라이선스:** CC BY-NC-SA (비상업적·동일조건변경허락)

> 1280×720 / ~1 FPS, 20일치 456,714 프레임, 4개 오피스 위치, bounding box +
> 행동 레이블(standing/sitting/talking/fallen) 포함. 본 실험에서는 약 78분
> 구간을 사용했다. 원본 영상은 라이선스상 본 저장소에 포함하지 않으며 위
> URL에서 받을 수 있다. 본 저장소에는 **분석 결과(그래프·로그·요약)**만 포함한다.

---

## 2. 실험 설정

| 항목 | 값 |
|---|---|
| 영상 길이 | 4,680초 (78분) |
| VLM 분석 포인트 | 156개 |
| 외기 조건 | 3°C 고정 (겨울철) |
| 재실 비율 | 90% (평균 1.3명, 최대 4명) |
| 초기 실내온도 | 10°C에서 가열 시작 |

**비교 방법:** 동일 영상을 두 제어기로 **동시 시뮬레이션**하여 공정 비교.

- **AI 제어** — VLM(옷차림·활동) + YOLO(인원) 인식 기반 동적 PMV 제어
- **룰베이스** — 재실 중 24°C 고정 + Fan2 상시 가동

에너지 모델(압축기 부하 기반): `P = P_fan + P_comp,rated × load`. 송풍 팬은
40/70/110 W(1/2/3단), 압축기 정격은 1,200 W로 두고, 부하율 `load`는 실내로
새어드는 열(외기 전도 + 재실 체열)을 에어컨 냉난방 능력으로 나눈 값이다. 설정온도까지
적극 구동하면 정격(load=1.0), 설정온도를 유지하면 열손실 상쇄분만큼의 부분부하
(서모스탯 사이클링)가 된다. 이 구조로 설정온도·외기조건이 전력에 자연히 반영된다.
부하율 계산에 쓰는 상수(열시상수 τ, 냉난방률, 체열)는 HVACSimulator와 동일하게
맞춰 물리와 전력 모델이 일관되도록 했다. 룰베이스는 재실 중 24°C 고정 + Fan2 상시.

---

## 3. 결과

| 지표 | AI 제어 | 룰베이스 |
|---|---|---|
| **총 에너지** | **254.33 Wh** | 799.85 Wh |
| **에너지 절감** | **−545.52 Wh (−68.2%)** | — |
| 쾌적율 (PMV ±0.5) | 85.9% | 84.6% |
| 평균 실내온도 | 21.62 °C | 22.97 °C |

### 핵심 해석
- **에너지 68.2% 절감** — AI는 쾌적 도달 시 OFF + 공실 OFF로 운전해 재실 중
  17%만 가동한 반면, 룰베이스는 재실 내내 24°C 유지를 위해 압축기를 상시 가동.
  겨울 옷차림(긴팔) 재실자는 약 22°C에서 이미 쾌적해, AI는 워밍업 후 대부분 정지.
- **쾌적도는 동등하나 질이 다름** — 정상상태(워밍업 후)에서 AI는 PMV를 약 +0.25
  (중립 근접)로, 룰베이스는 +0.49(더움 경계)로 유지. **덜 쓰면서 더 안정적인 쾌적**.

---

## 4. 그래프

| 파일 | 내용 |
|---|---|
| [01_pmv_comparison.png](experiment/01_pmv_comparison.png) | PMV 시계열 비교 (AI vs 룰베이스) |
| [02_indoor_temp.png](experiment/02_indoor_temp.png) | 실내온도 비교 |
| [03_energy_cumulative.png](experiment/03_energy_cumulative.png) | 누적 에너지 소비 |
| [04_energy_bar.png](experiment/04_energy_bar.png) | 총 에너지 절감 막대 |
| [05_comfort_rate.png](experiment/05_comfort_rate.png) | 쾌적율 비교 |
| [06_activity_distribution.png](experiment/06_activity_distribution.png) | VLM 활동 분류 분포 |

원자료: [analysis_log.csv](experiment/analysis_log.csv) (프레임 단위 전체 로그) ·
[summary.txt](experiment/summary.txt) (요약 수치)

---

## 5. 한계 (정직한 평가)

이 영상은 전원이 겨울 긴팔·착석으로 **균일**해, VLM의 옷차림 '적응성'보다 제어
알고리즘(동적 목표온도·공실 OFF·쾌적 사이클링)의 효과가 부각된 케이스다.
옷차림 변화에 대한 실시간 적응은 별도 시연 영상으로 입증했다
([데모 영상](demo/vlm_hvac_demo.mp4) — 동일 인물이 민소매→반팔→아우터로 바꿀 때
clo·PMV·냉난방이 자동으로 바뀌는 모습).
