import math


class ThermalEngine:
    """
    [열쾌적성 판단 엔진]
    출처: ISO 7730:2005 - Ergonomics of the thermal environment
           ASHRAE Standard 55 - Thermal Environmental Conditions for Human Occupancy
    설명: PMV(Predicted Mean Vote) 모델을 기반으로 사용자의 온열감을 예측합니다.
    """

    def __init__(self):
        self.STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

    def calculate_pmv(self, ta, tr, rh, vel, met, clo):
        """
        PMV 지수 계산 (ISO 7730:2005 기반)

        Args:
            ta  : 실내 공기 온도 (°C)
            tr  : 평균 복사 온도 (°C)  – 보통 ta와 동일하게 근사
            rh  : 상대 습도 (%)
            vel : 기류 속도 (m/s)
            met : 대사율 (met)  – 1.0=착석, 1.5=보행, 3.0=운동
            clo : 착의량 (clo)  – 0.5=반팔, 1.0=긴팔, 1.3=긴팔+아우터

        Returns:
            float: PMV 지수 [-3.0, 3.0] (소수점 둘째 자리)
        """
        # 1. 입력값 유효 범위 강제 (비정상 입력 방어)
        ta  = max(10.0, min(40.0, float(ta)))
        tr  = max(10.0, min(50.0, float(tr)))
        rh  = max(0.0,  min(100.0, float(rh)))
        vel = max(0.0,  min(5.0,   float(vel)))
        met = max(0.8,  min(4.0,   float(met)))
        clo = max(0.0,  min(2.0,   float(clo)))

        # 2. 기초 데이터 변환
        # 수증기 분압 (Pa) — ISO 7730 Annex D: pa = RH[%] × 10 × exp(16.6536 − 4030.183/(ta+235))
        # 뒤의 열손실 식 상수(5733, 5867)가 Pa 기준이므로 반드시 Pa로 계산해야 한다.
        # (rh/100로 쓰면 100배 작아져 증발 열손실이 과대평가 → 실제보다 시원하게 나옴)
        pa  = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))
        if not math.isfinite(pa):
            pa = 0.0

        icl = 0.155 * clo   # 착의 열저항 (m²·K/W)
        m   = met * 58.15   # 대사율 (W/m²)
        mw  = m              # 외부 작업 = 0

        # 3. 착의 표면적 계수
        fcl = 1.00 + 1.290 * icl if icl <= 0.078 else 1.05 + 0.645 * icl

        # 4. 착의 표면 온도(tcl) 반복 수렴
        tcl = ta + (35.5 - ta) / (3.5 * (6.45 * icl + 0.1))
        tcl = max(10.0, min(50.0, tcl))  # 초기값 안전 클램핑

        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p5 = 308.7 - 0.028 * mw + p2 * ((tr + 273) / 100) ** 4

        hc = max(2.38 * abs(tcl - ta) ** 0.25, 12.1 * math.sqrt(vel))

        for _ in range(150):
            hc = max(2.38 * abs(tcl - ta) ** 0.25, 12.1 * math.sqrt(vel))

            denom = 1.0 + p1 * hc
            if abs(denom) < 1e-10:
                break

            tcl_new = (p5 + p3 * hc - p2 * ((tcl + 273) / 100) ** 4) / denom

            if not math.isfinite(tcl_new) or not (-50 < tcl_new < 150):
                break

            if abs(tcl - tcl_new) < 0.0001:
                tcl = tcl_new
                break
            tcl = tcl_new

        # 루프 후 hc 최솟값 보장 (vel=0일 때 hc=0 방지)
        hc = max(hc, 12.1 * math.sqrt(vel), 0.5)

        # 5. 열부하(L) 계산
        hl_data = self._compute_heat_loads(mw, pa, ta, tcl, tr, fcl, hc)
        hl_sum = sum(hl_data.values())

        ts = 0.303 * math.exp(-0.036 * m) + 0.028
        if not math.isfinite(ts):
            return 0.0

        pmv = ts * (mw - hl_sum)

        if not math.isfinite(pmv):
            return 0.0

        return round(max(-3.0, min(3.0, pmv)), 2)

    def _compute_heat_loads(self, mw, pa, ta, tcl, tr, fcl, hc):
        """열손실 항목별 계산 (W/m²) - ISO 7730:2005 Table C.1 기준"""
        hl_skin     = 3.05e-3 * (5733 - 6.99 * mw - pa)
        hl_sweat    = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
        hl_res_lat  = 1.7e-5 * mw * (5867 - pa)
        hl_res_dry  = 0.0014 * mw * (34 - ta)
        hl_rad      = 3.96 * fcl * (((tcl + 273) / 100) ** 4 - ((tr + 273) / 100) ** 4)
        hl_conv     = fcl * hc * (tcl - ta)

        result = {
            'skin':     hl_skin     if math.isfinite(hl_skin)     else 0.0,
            'sweat':    hl_sweat    if math.isfinite(hl_sweat)    else 0.0,
            'res_lat':  hl_res_lat  if math.isfinite(hl_res_lat)  else 0.0,
            'res_dry':  hl_res_dry  if math.isfinite(hl_res_dry)  else 0.0,
            'rad':      hl_rad      if math.isfinite(hl_rad)      else 0.0,
            'conv':     hl_conv     if math.isfinite(hl_conv)     else 0.0,
        }
        return result

    # ── 기류 연동 (팬속도 → 체감 기류속도) ──────────────────────────────────
    # 냉난방 가동 중에는 실제 기류가 생기므로 PMV에 반영해야 한다.
    # 정지공기(0.1)로 고정하면 "냉방 중인데 정지공기로 계산"하는 모순이 생겨
    # 실제보다 덥게(냉방 시) 평가되어 조기 정지·재가동이 반복된다.
    # 값은 '재실자 위치'의 기류 기준. 토출구 풍속이 아니라 확산 후 체감값이므로
    # 공조 중인 실내 거주역은 통상 0.1~0.25 m/s 수준이다. 과도하게 잡으면
    # 켜자마자 PMV가 급락해 즉시 정지하는 on/off 진동이 발생한다.
    FAN_VEL = {0: 0.10, 1: 0.15, 2: 0.20, 3: 0.25}   # m/s

    @staticmethod
    def air_velocity(hvac_on: bool, fan_speed: int) -> float:
        """가동 상태·팬속도에 따른 체감 기류속도(m/s)."""
        if not hvac_on:
            return ThermalEngine.FAN_VEL[0]
        return ThermalEngine.FAN_VEL.get(int(fan_speed), 0.30)

    # ── 외기 적응형 보정 (ASHRAE 55 adaptive comfort 개념) ──────────────────
    # 사람은 바깥 날씨에 적응한다: 한여름엔 더 시원한 실내를, 한겨울엔 더 따뜻한
    # 실내를 기대한다. PMV는 이를 반영하지 못하므로 목표 PMV 기준선을 이동시킨다.
    ADAPT_SUMMER_T = 26.0   # 이 외기 이상이면 여름 모드
    ADAPT_WINTER_T = 12.0   # 이 외기 이하면 겨울 모드
    ADAPT_MAX_SHIFT = 0.4   # 목표 PMV 최대 이동폭

    @classmethod
    def adaptive_pmv_offset(cls, outdoor_temp) -> float:
        """외기온에 따른 목표 PMV 오프셋.

        여름(더울수록) → 음수: 더 시원한 상태를 '중립'으로 삼아 냉방을 더 오래 유지
        겨울(추울수록) → 양수: 더 따뜻한 상태를 '중립'으로 삼아 난방을 더 오래 유지
        """
        if outdoor_temp is None:
            return 0.0
        t = float(outdoor_temp)
        if t >= cls.ADAPT_SUMMER_T:
            over = min(t - cls.ADAPT_SUMMER_T, 10.0)      # 26~36°C 구간에서 선형
            return -cls.ADAPT_MAX_SHIFT * (over / 10.0)
        if t <= cls.ADAPT_WINTER_T:
            under = min(cls.ADAPT_WINTER_T - t, 12.0)     # 12~0°C 구간에서 선형
            return cls.ADAPT_MAX_SHIFT * (under / 12.0)
        return 0.0

    def get_comfort_status(self, pmv):
        """PMV 수치에 따른 상태 텍스트 반환 (한국어/영문)"""
        if   pmv >  2.5: return "매우 더움 (Hot)"
        elif pmv >  1.5: return "더움 (Warm)"
        elif pmv >  0.5: return "조금 더움 (Slightly Warm)"
        elif pmv > -0.5: return "쾌적 (Neutral)"
        elif pmv > -1.5: return "조금 추움 (Slightly Cool)"
        elif pmv > -2.5: return "추움 (Cool)"
        else:            return "매우 추움 (Cold)"
