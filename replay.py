"""
오프라인 제어 재생기 — 기존 analysis_log.csv의 '인식 결과'(clo/met/인원/외기)를
그대로 재사용하고, 제어·물리·에너지만 다시 계산한다.
VLM/YOLO를 다시 돌리지 않으므로 6시간 영상이 수 초에 끝난다.

전제: 인식 결과는 HVAC 상태와 무관하다(카메라 영상만 보므로) → 재생이 타당함.
"""
import sys, argparse
import pandas as pd

from thermal_engine import ThermalEngine
from hvac_simulator import HVACSimulator
from pid_controller import PIDController
from vlm_processor import VLMProcessor
import control_logic as CL
from control_logic import decide_control
from energy_monitor import hvac_watts, rb_mode_setpoint, RB_FAN, RB_COOL_SETPOINT
from state_machine import StateManager

CTRL_SEC = 5.0


def replay(csv_path, fan_vel0=None, pmv_off=None, min_cycle=None,
           init_temp=None, init_humid=None, room_m2=30.0, verbose=False,
           cool_targets=None, comfort_temp=None, min_fan=None,
           legacy_humid=False, hold_target=False, fan_only=False,
           rb_setpoint=None, smooth=False):
    d = pd.read_csv(csv_path)
    if cool_targets is not None:
        CL._COOL_TARGETS = list(cool_targets)
    if comfort_temp is not None:
        CL.COMFORT_TEMP = comfort_temp

    # ── 파라미터 주입 ──
    if fan_vel0 is not None:
        ThermalEngine.FAN_VEL = dict(ThermalEngine.FAN_VEL)
        ThermalEngine.FAN_VEL[0] = fan_vel0
    if pmv_off is not None:
        CL.PMV_OFF = pmv_off

    if legacy_humid:      # 수정 전 습도 모델 재현(단방향 감소 + 15% 하한)
        import hvac_simulator as HS, math as _m
        def _old(self, outdoor_temp, outdoor_humid=50.0, people_count=0, dt=1/30):
            sf = 20.0 / max(self.room_size, 5.0)
            tau = self.TAU_BUILDING/5.0 if self.window_open else self.TAU_BUILDING
            self.indoor_temp += (outdoor_temp - self.indoor_temp)*(1-_m.exp(-dt/tau))
            if self.is_on:
                if self.mode=='heat' and self.indoor_temp < self.target_temp:
                    self.indoor_temp = min(self.target_temp, self.indoor_temp + self.HEAT_RATE*self.fan_speed*sf*dt)
                    self.indoor_humid = max(25.0, self.indoor_humid - 0.03*dt)
                elif self.mode=='cool' and self.indoor_temp > self.target_temp:
                    self.indoor_temp = max(self.target_temp, self.indoor_temp - self.COOL_RATE*self.fan_speed*sf*dt)
                    self.indoor_humid -= 0.09*dt
            if self.window_open:
                self.indoor_humid += (outdoor_humid - self.indoor_humid)*(dt/300.0)
            self.indoor_temp += people_count*self.BODY_HEAT*sf*dt
            self.indoor_temp = max(10.0, min(45.0, self.indoor_temp))
            self.indoor_humid = max(15.0, min(95.0, self.indoor_humid))
            return round(self.indoor_temp,2), round(self.indoor_humid,1)
        HS.HVACSimulator.simulate_step = _old

    engine = ThermalEngine()
    ai = HVACSimulator(room_size=room_m2)
    rb = HVACSimulator(room_size=room_m2)
    ai.indoor_temp = init_temp if init_temp is not None else float(d.ai_indoor_temp.iloc[0])
    rb.indoor_temp = init_temp if init_temp is not None else float(d.rb_indoor_temp.iloc[0])
    ai.indoor_humid = init_humid if init_humid is not None else float(d.ai_indoor_humid.iloc[0])
    rb.indoor_humid = init_humid if init_humid is not None else float(d.rb_indoor_humid.iloc[0])

    pid = PIDController()
    sm = StateManager()

    ai_wh = rb_wh = 0.0
    comfort_n = total_n = 0
    toggles = 0
    last_on = ai.is_on
    last_switch_t = -1e9          # 최소 가동/정지 시간용
    rows = []

    # VLM 결과 평활(중앙값+변화율 제한) — vlm_processor._smooth 와 동일 로직
    if smooth:
        from vlm_processor import VLMProcessor as VP
        import statistics
        ch, mh, cs, ms = [], [], [None], [None]
        sm_clo, sm_met = {}, {}
        for _i in range(len(d)):
            _c = float(d.ai_clo.iloc[_i]) if pd.notna(d.ai_clo.iloc[_i]) else 0.7
            _mv = float(d.ai_met.iloc[_i]) if pd.notna(d.ai_met.iloc[_i]) else 1.2
            ch.append(_c); mh.append(_mv)
            if len(ch) > VP.SMOOTH_WINDOW: ch.pop(0); mh.pop(0)
            cmed, mmed = statistics.median(ch), statistics.median(mh)
            if cs[0] is None:
                cs[0], ms[0] = cmed, mmed
            else:
                cs[0] += max(-VP.CLO_MAX_STEP, min(VP.CLO_MAX_STEP, cmed - cs[0]))
                ms[0] += max(-VP.MET_MAX_STEP, min(VP.MET_MAX_STEP, mmed - ms[0]))
            sm_clo[_i], sm_met[_i] = round(cs[0], 2), round(ms[0], 2)
        d = d.copy()
        d["ai_clo"] = [sm_clo[i] for i in range(len(d))]
        d["ai_met"] = [sm_met[i] for i in range(len(d))]

    t_end = float(d.video_time_sec.iloc[-1])
    idx = 0
    t = float(d.video_time_sec.iloc[0])

    while t <= t_end:
        # ── 이 시점의 '인식 결과' (30초 로그를 계단 유지) ──
        while idx + 1 < len(d) and float(d.video_time_sec.iloc[idx + 1]) <= t:
            idx += 1
        r = d.iloc[idx]
        people = int(r.people_count)
        clo = float(r.ai_clo) if pd.notna(r.ai_clo) else 0.7
        met = float(r.ai_met) if pd.notna(r.ai_met) else 1.2
        heat = str(r.vlm_heat_source)
        out_t = float(r.outdoor_temp)
        out_h = float(r.outdoor_humid)

        # ── 물리 1스텝 (0.1s 세분) ──
        nsub = max(1, int(CTRL_SEC / 0.1))
        sdt = CTRL_SEC / nsub
        ai.set_room(room_m2, False)
        rb.set_room(room_m2, False)
        for _ in range(nsub):
            ai.simulate_step(out_t, out_h, people, dt=sdt)
            rb.simulate_step(out_t, out_h, people, dt=sdt)

        # 측정 시점(제어 적용 전) 공조 상태 — CSV 정합성용
        meas_on, meas_fan = ai.is_on, ai.fan_speed
        meas_tgt, meas_mode = ai.target_temp, ai.mode
        rmeas_on, rmeas_fan, rmeas_tgt = rb.is_on, rb.fan_speed, rb.target_temp

        # ── PMV (현재 상태 기준 — main.py와 동일 순서) ──
        tr = ai.indoor_temp + (VLMProcessor.TR_HEAT_OFFSET if heat == "yes" else 0.0)
        adapt = engine.adaptive_pmv_offset(out_t)
        if people > 0:
            vel = engine.air_velocity(ai.is_on, ai.fan_speed, getattr(ai,'fan_only',False))
            pmv = engine.calculate_pmv(ai.indoor_temp, tr, ai.indoor_humid,
                                       vel, met, clo) - adapt
            rvel = engine.air_velocity(rb.is_on, rb.fan_speed)
            rpmv = engine.calculate_pmv(rb.indoor_temp, rb.indoor_temp,
                                        rb.indoor_humid, rvel, met, clo) - adapt
            total_n += 1
            if -0.5 < pmv < 0.5:
                comfort_n += 1
        else:
            pmv = rpmv = None

        # ── 제어 결정 ──
        sm.update(people, str(r.vlm_outerwear), str(r.vlm_activity))
        power, tgt, fan, mode = decide_control(
            pmv if pmv is not None else 0.0, people, pid,
            ai.is_on, ai.mode, dt=CTRL_SEC, current_fan=ai.fan_speed,
            indoor_temp=ai.indoor_temp, state=sm.state.value,
            clo=clo, outdoor_temp=out_t)
        # 히스테리시스 구간에서 켜질 때 잡은 목표를 유지(24°C로 되돌리지 않음)
        if (hold_target and power is True and mode == "cool" and ai.is_on
                and ai.target_temp < tgt):
            tgt = ai.target_temp

        # ── ③ 최소 가동/정지 시간 ──
        if min_cycle and power is not None and power != ai.is_on:
            if t - last_switch_t < min_cycle:
                power = None                      # 전환 보류
            else:
                last_switch_t = t

        if min_fan is not None and power is True:
            fan = max(fan, min_fan)
        if power is True:
            ai.set_control(power=True, target=tgt, fan=fan, mode=mode)
        elif power is False:
            _fo = (mode == "fan_only")       # 제어로직이 결정한 송풍 유지 여부
            ai.set_control(power=False, target=ai.target_temp,
                           fan=(fan if _fo else 1), fan_only=_fo)

        if ai.is_on != last_on:
            toggles += 1
            last_on = ai.is_on

        # ── 룰베이스 제어 ──
        if people > 0:
            rmode, rset = rb_mode_setpoint(out_t)
            if rb_setpoint is not None and rmode == "cool":
                rset = rb_setpoint          # 베이스라인 설정온도 변경(비교 실험용)
            rb.set_control(power=True, target=rset, fan=RB_FAN, mode=rmode)
        else:
            rb.set_control(power=False, target=RB_COOL_SETPOINT, fan=1)

        # ── 에너지 적산 ──
        dt_h = CTRL_SEC / 3600.0
        ai_wh += hvac_watts(ai.is_on, ai.indoor_temp, ai.target_temp, ai.fan_speed,
                            ai.mode, ai.room_size, out_t, people,
                            fan_only=getattr(ai,'fan_only',False)) * dt_h
        rb_wh += hvac_watts(rb.is_on, rb.indoor_temp, rb.target_temp, rb.fan_speed,
                            rb.mode, rb.room_size, out_t, people) * dt_h

        rows.append(dict(
            video_time_sec=round(t,2), frame_number=int(t*30),
            vlm_activity=str(r.vlm_activity), vlm_sleeves=str(r.vlm_sleeves),
            vlm_outerwear=str(r.vlm_outerwear), vlm_room_size=str(r.vlm_room_size),
            vlm_heat_source=heat, vlm_raw="",
            outdoor_temp=out_t, outdoor_humid=out_h, people_count=people,
            ai_clo=clo, ai_met=met, ai_pmv=pmv,
            ai_comfort_status=(engine.get_comfort_status(pmv) if pmv is not None else "-"),
            ai_hvac_on=meas_on, ai_target_temp=meas_tgt,
            ai_fan_speed=meas_fan, ai_mode=meas_mode,
            ai_indoor_temp=round(ai.indoor_temp,2), ai_indoor_humid=round(ai.indoor_humid,1),
            ai_energy_wh=round(ai_wh,4),
            rb_clo=clo, rb_met=met, rb_pmv=rpmv,
            rb_comfort_status=(engine.get_comfort_status(rpmv) if rpmv is not None else "-"),
            rb_hvac_on=rmeas_on, rb_target_temp=rmeas_tgt,
            rb_fan_speed=rmeas_fan, rb_indoor_temp=round(rb.indoor_temp,2),
            rb_indoor_humid=round(rb.indoor_humid,1), rb_energy_wh=round(rb_wh,4),
            # 내부 분석용
            t=t, people=people, pmv=pmv, rpmv=rpmv,
            ai_temp=ai.indoor_temp, rb_temp=rb.indoor_temp, on=ai.is_on, fan=ai.fan_speed))
        t += CTRL_SEC

    rb_comfort = sum(1 for x in rows if x["rpmv"] is not None and -0.5 < x["rpmv"] < 0.5)
    rb_total = sum(1 for x in rows if x["rpmv"] is not None)

    return dict(
        ai_wh=ai_wh, rb_wh=rb_wh,
        savings=(rb_wh - ai_wh) / rb_wh * 100 if rb_wh else 0,
        comfort=comfort_n / total_n * 100 if total_n else 0,
        rb_comfort=rb_comfort / rb_total * 100 if rb_total else 0,
        toggles=toggles,
        ai_temp=sum(x["ai_temp"] for x in rows) / len(rows),
        rb_temp=sum(x["rb_temp"] for x in rows) / len(rows),
        pmv_median=pd.Series([x["pmv"] for x in rows if x["pmv"] is not None]).median(),
        on_rate=sum(1 for x in rows if x["people"] > 0 and x["on"]) /
                max(1, sum(1 for x in rows if x["people"] > 0)) * 100,
        rows=rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--fan-vel0", type=float, default=None)
    ap.add_argument("--pmv-off", type=float, default=None)
    ap.add_argument("--min-cycle", type=float, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="재생 결과 CSV 저장 경로")
    a = ap.parse_args()
    r = replay(a.csv, a.fan_vel0, a.pmv_off, a.min_cycle)
    if a.out:
        cols = ["video_time_sec","frame_number","vlm_activity","vlm_sleeves",
                "vlm_outerwear","vlm_room_size","vlm_heat_source","vlm_raw",
                "outdoor_temp","outdoor_humid","people_count","ai_clo","ai_met",
                "ai_pmv","ai_comfort_status","ai_hvac_on","ai_target_temp",
                "ai_fan_speed","ai_mode","ai_indoor_temp","ai_indoor_humid",
                "ai_energy_wh","rb_clo","rb_met","rb_pmv","rb_comfort_status",
                "rb_hvac_on","rb_target_temp","rb_fan_speed","rb_indoor_temp",
                "rb_indoor_humid","rb_energy_wh"]
        import os
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pd.DataFrame(r["rows"])[cols].to_csv(a.out, index=False)
        print("CSV 저장:", a.out)
    print(f"{a.label:24} 쾌적 {r['comfort']:5.1f}%  절감 {r['savings']:5.1f}%  "
          f"PMV중앙 {r['pmv_median']:+.2f}  전환 {r['toggles']:4d}회  "
          f"가동률 {r['on_rate']:5.1f}%  실내 {r['ai_temp']:.2f}°C")
