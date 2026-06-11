from __future__ import annotations

"""
HVAC 시나리오 시뮬레이터
────────────────────────────────────────────────────────────────────────────
카메라·VLM·날씨 API 없이 시나리오 JSON 으로 24시간 제어 로직을 고속 검증합니다.

  ─ HVACSimulator  : 실내 온습도 물리 시뮬레이션
  ─ ThermalEngine  : ISO 7730 PMV 계산
  ─ PIDController  : 팬 속도 제어 (dt 명시 전달)
  ─ decide_control : PMV 기반 히스테리시스 제어
  ─ decide_window  : 창문 개폐 판단
  ─ _SimStateTracker : 시뮬레이션 프레임 기반 상태 추적 (EMPTY/ARRIVAL/STEADY/PRE_DEPARTURE)

사용법:
  # 특정 시나리오 1개
  python scenario_runner.py --scenario scenarios/summer_office.json

  # scenarios/ 전체 일괄 실행
  python scenario_runner.py

  # 결과 폴더 지정
  python scenario_runner.py --output-dir my_results/

결과:
  results/<시나리오명>.csv   ── 1분 단위 로그 (1440행, state + solution 컬럼 포함)
  results/<시나리오명>.png   ── 3-패널 시각화 (PMV / 온도 / HVAC 상태)
"""

import argparse
import io
import json
import platform
import sys
from pathlib import Path

# Windows 터미널 한글 출력 보장
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from hvac_simulator import HVACSimulator
from thermal_engine  import ThermalEngine
from pid_controller  import PIDController
from control_logic   import decide_control, decide_window, FAN_VELOCITY, TR_HEAT_OFFSET

# ── 시뮬레이션 상수 ────────────────────────────────────────────────────────────
FPS               = 30
TOTAL_SIM_HOURS   = 24
TOTAL_FRAMES      = TOTAL_SIM_HOURS * 3600 * FPS   # 2,592,000 프레임
PMV_UPDATE_FRAMES = 5  * FPS        # PMV 재계산·제어 결정: 5 sim-초마다
LOG_FRAMES        = 60 * FPS        # 로그 기록: 1 sim-분마다 (→ 1440행/일)
PROGRESS_FRAMES   = 3600 * FPS      # 콘솔 진행 출력: 1 sim-시간마다


# ── 시뮬레이션 상태 추적기 ────────────────────────────────────────────────────

class _SimStateTracker:
    """
    실시간 타이머 대신 시뮬레이션 프레임 수 기반 상태 관리.
    StateManager 와 동일한 전이 규칙을 시뮬레이션 시간으로 재구현.

    EMPTY → ARRIVAL    : 인원 최초 감지
    ARRIVAL → STEADY   : 10 sim-분 경과
    STEADY → PRE_DEPARTURE : 퇴근 맥락 점수 ≥ 55
    ANY → EMPTY        : 인원 0 상태 30 sim-초 지속
    """

    ARRIVAL_HOLD_FRAMES  = 10 * 60 * FPS   # 10 sim-분
    EMPTY_CONFIRM_FRAMES = 30 * FPS         # 30 sim-초

    def __init__(self, work_end_hour: int = 18):
        self.state            = "EMPTY"
        self._work_end_hour   = work_end_hour
        self._prev_people     = 0
        self._arrival_frame   = 0
        self._empty_since     = None

    def update(self, frame_idx: int, sim_sec: int,
               people: int, clo: float = 1.0, met: float = 1.2) -> str:
        """
        매 PMV_UPDATE_FRAMES마다 호출.

        Args:
            frame_idx : 현재 시뮬레이션 프레임 번호
            sim_sec   : 시뮬레이션 경과 초 (0 ~ 86400)
            people    : 현재 인원 수
            clo       : 착의량 (clo ≥ 1.3 → 아우터 착용 간주)
            met       : 대사율 (1.2 ≤ met < 1.5 → 기립 간주)
        """
        hour          = sim_sec // 3600
        has_outerwear = clo >= 1.3
        is_standing   = 1.2 <= met < 1.5

        # ── 인원 0 지속 → EMPTY ───────────────────────────────────────────────
        if people == 0:
            if self._empty_since is None:
                self._empty_since = frame_idx
            elif frame_idx - self._empty_since >= self.EMPTY_CONFIRM_FRAMES:
                if self.state != "EMPTY":
                    print(f"  [SimState] {self.state} → EMPTY")
                self.state          = "EMPTY"
                self._arrival_frame = 0
        else:
            self._empty_since = None

        # ── 상태별 전이 ───────────────────────────────────────────────────────
        if self.state == "EMPTY":
            if people > 0:
                print(f"  [SimState] EMPTY → ARRIVAL  ({sim_sec//3600:02d}:{(sim_sec%3600)//60:02d})")
                self.state          = "ARRIVAL"
                self._arrival_frame = frame_idx

        elif self.state == "ARRIVAL":
            if frame_idx - self._arrival_frame >= self.ARRIVAL_HOLD_FRAMES:
                print(f"  [SimState] ARRIVAL → STEADY  ({sim_sec//3600:02d}:{(sim_sec%3600)//60:02d})")
                self.state = "STEADY"

        elif self.state == "STEADY":
            score = 0
            if people < self._prev_people: score += 30
            if has_outerwear:              score += 25
            if is_standing:               score += 10
            if abs(hour - self._work_end_hour) <= 1: score += 10
            if score >= 55:
                print(f"  [SimState] STEADY → PRE_DEPARTURE  (score={score})")
                self.state = "PRE_DEPARTURE"

        elif self.state == "PRE_DEPARTURE":
            score = 0
            if people < self._prev_people: score += 30
            if has_outerwear:              score += 25
            if is_standing:               score += 10
            if abs(hour - self._work_end_hour) <= 1: score += 10
            if score < 30:
                self.state = "STEADY"

        self._prev_people = people
        return self.state


# ── 솔루션 텍스트 생성 ────────────────────────────────────────────────────────

def _solution_text(state: str, pmv: float, hvac_on: bool,
                   hvac_mode: str, fan_speed: int, target_temp: float,
                   indoor_temp: float,
                   window_rec: bool | None = None) -> str:
    """
    현재 상태·PMV 기반 솔루션 텍스트 반환.

    window_rec : decide_window() 반환값
        True  → '창문 열기 권장'  (환기 or 자연 냉방 보조)
        False → '창문 닫기 권장'  (냉난방 효율 유지)
        None  → 창문 언급 없음    (현재 상태 유지)

    ※ 창문 권장은 솔루션 텍스트에만 반영되며,
       실내 온도 물리 계산에는 영향을 주지 않습니다.
    """
    if state == "EMPTY":
        return "공실 감지 - 에어컨 OFF"

    if state == "ARRIVAL":
        mode_str = "난방" if hvac_mode == "heat" else "냉방"
        base = f"도착 감지 - {mode_str} 강화 (목표 {target_temp:.0f}C)"
    elif state == "PRE_DEPARTURE":
        base = "퇴근 준비 맥락 감지 - 절전 모드 전환"
    elif not hvac_on:
        if abs(pmv) <= 0.2:
            base = f"PMV {pmv:+.2f} - 쾌적, 에어컨 OFF"
        elif pmv > 0:
            base = f"PMV {pmv:+.2f} - 더움 감지, 냉방 준비"
        else:
            base = f"PMV {pmv:+.2f} - 추움 감지, 난방 준비"
    else:
        mode_str  = "냉방" if hvac_mode == "cool" else "난방"
        intensity = "약" if fan_speed == 1 else "중" if fan_speed == 2 else "강"
        base = (f"PMV {pmv:+.2f} - {mode_str}({intensity}) Fan{fan_speed}"
                f"  {indoor_temp:.1f}C -> {target_temp:.0f}C")

    # 창문 권장 사항 (솔루션 텍스트 부가 정보, 물리 미반영)
    if window_rec is True:
        return base + " | 창문 열기 권장"
    if window_rec is False:
        return base + " | 창문 닫기 권장"
    return base


# ── 유틸 함수 ──────────────────────────────────────────────────────────────────

def _hhmm_to_sec(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 3600 + m * 60


def _current_slot(timeline: list, sim_sec: int) -> dict:
    slot = timeline[0]
    for s in timeline:
        if sim_sec >= _hhmm_to_sec(s["start"]):
            slot = s
    return slot


def _setup_korean_font():
    candidates = {
        "Darwin":  ["AppleGothic", "Apple SD Gothic Neo"],
        "Windows": ["Malgun Gothic", "맑은 고딕"],
        "Linux":   ["NanumGothic", "NanumBarunGothic", "UnDotum"],
    }.get(platform.system(), [])
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return
    plt.rcParams["axes.unicode_minus"] = False


# ── 그래프 생성 ───────────────────────────────────────────────────────────────

def _plot(df: pd.DataFrame, title: str, save_path: Path):
    _setup_korean_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    x = range(len(df))

    # 패널 1: PMV
    ax1 = axes[0]
    ax1.axhspan(-0.5,  0.5, alpha=0.13, color="limegreen", zorder=0)
    ax1.axhspan( 0.5,  1.5, alpha=0.09, color="yellow",    zorder=0)
    ax1.axhspan(-1.5, -0.5, alpha=0.09, color="skyblue",   zorder=0)
    ax1.axhspan( 1.5,  3.0, alpha=0.09, color="orange",    zorder=0)
    ax1.axhspan(-3.0, -1.5, alpha=0.09, color="plum",      zorder=0)
    ax1.axhline(0,    color="gray",  lw=0.8, linestyle="--", zorder=1)
    ax1.axhline( 0.5, color="olive", lw=0.5, linestyle=":",  zorder=1)
    ax1.axhline(-0.5, color="olive", lw=0.5, linestyle=":",  zorder=1)
    ax1.plot(x, df["pmv"], color="crimson", lw=1.7, label="PMV", zorder=2)
    ax1.set_ylim(-3.0, 3.0)
    ax1.set_ylabel("PMV 지수")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.2)

    # 패널 2: 온도
    ax2 = axes[1]
    ax2.plot(x, df["indoor_temp"],  color="tomato",    lw=1.7, label="실내온도")
    ax2.plot(x, df["outdoor_temp"], color="steelblue", lw=1.0, ls="--", label="실외온도")
    ax2.plot(x, df["target_temp"],  color="seagreen",  lw=0.9, ls=":",  label="목표온도 (24°C)")
    ax2.set_ylabel("온도 (°C)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.2)

    # 패널 3: HVAC 상태 + 인원
    ax3 = axes[2]
    cool_mask = df["hvac_on"] & (df["hvac_mode"] == "cool")
    heat_mask = df["hvac_on"] & (df["hvac_mode"] == "heat")
    off_mask  = ~(cool_mask | heat_mask)
    fan = df["fan_speed"].astype(float)

    ax3.fill_between(x, 0, fan.where(cool_mask, 0),
                     step="post", color="#3399ff", alpha=0.75, label="냉방")
    ax3.fill_between(x, 0, fan.where(heat_mask, 0),
                     step="post", color="#ff5533", alpha=0.75, label="난방")
    ax3.fill_between(x, 0, 0.25,
                     where=off_mask.values,
                     step="post", color="#bbbbbb", alpha=0.55, label="꺼짐")
    ax3.set_ylim(0, 3.8)
    ax3.set_yticks([1, 2, 3])
    ax3.set_ylabel("팬 속도")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(axis="y", alpha=0.2)

    ax3b = ax3.twinx()
    ax3b.plot(x, df["people"], color="black", lw=1.2, ls="-.", alpha=0.5, label="인원(명)")
    ax3b.set_ylabel("인원 (명)", fontsize=8)
    ax3b.set_ylim(0, max(df["people"].max() * 2, 2))
    ax3b.legend(loc="upper right", fontsize=8)

    tick_idx = list(range(0, len(df), 120))
    ax3.set_xticks(tick_idx)
    ax3.set_xticklabels(
        [df["time"].iloc[i] for i in tick_idx],
        rotation=45, fontsize=8,
    )
    ax3.set_xlabel("시각")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 시나리오 실행 ─────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, output_dir: Path) -> pd.DataFrame:
    name = scenario["name"]
    print(f"\n{'=' * 62}")
    print(f"  {name}")
    if "description" in scenario:
        desc = scenario["description"]
        print(f"  {desc[:75]}{'...' if len(desc) > 75 else ''}")
    print(f"{'=' * 62}")

    # ── 초기화 ────────────────────────────────────────────────────────────────
    room_m2 = scenario.get("room_size_m2", 30.0)
    hvac    = HVACSimulator(room_size=room_m2)
    hvac.indoor_temp  = scenario.get("initial_indoor_temp",  22.0)
    hvac.indoor_humid = scenario.get("initial_indoor_humid", 50.0)
    hvac.set_room(room_m2, False)

    engine   = ThermalEngine()
    pid      = PIDController(kp=0.8, ki=0.05, kd=0.3)
    sim_state = _SimStateTracker(work_end_hour=scenario.get("work_end_hour", 18))

    outdoor_base = scenario["outdoor"]
    timeline = sorted(scenario["timeline"], key=lambda s: _hhmm_to_sec(s["start"]))

    logs      = []
    pmv_val   = 0.0
    window_rec: bool | None = None   # 창문 권장 (솔루션 텍스트 전용)

    # ── 메인 시뮬레이션 루프 ─────────────────────────────────────────────────
    for frame_idx in range(TOTAL_FRAMES):
        sim_sec_int = frame_idx // FPS

        slot      = _current_slot(timeline, sim_sec_int)
        out_temp  = slot.get("outdoor_temp",  outdoor_base["temp"])
        out_humid = slot.get("outdoor_humid", outdoor_base["humid"])
        people    = slot["people"]
        met       = slot.get("met", 1.2)
        clo       = slot.get("clo", 1.0)
        heat_src  = "yes" if slot.get("heat_source", False) else "no"

        # HVAC 물리 시뮬레이션 (30fps 속도)
        hvac.simulate_step(out_temp, out_humid, people_count=people)

        # PMV 재계산 + 제어 결정 (5 sim-초마다)
        if frame_idx % PMV_UPDATE_FRAMES == 0:
            air_vel = FAN_VELOCITY.get(hvac.fan_speed, 0.1)
            tr      = hvac.indoor_temp + (TR_HEAT_OFFSET if heat_src == "yes" else 0.0)
            pmv_val = engine.calculate_pmv(
                ta=hvac.indoor_temp, tr=tr,
                rh=hvac.indoor_humid, vel=air_vel,
                met=met, clo=clo,
            )

            # 상태 머신 업데이트
            current_state = sim_state.update(
                frame_idx, sim_sec_int, people, clo=clo, met=met)

            power, tgt, fan, mode = decide_control(
                pmv_val, people, pid, hvac.is_on, hvac.mode,
                dt=5.0, current_fan=hvac.fan_speed)
            if power:
                hvac.set_control(power=True, target=tgt, fan=fan, mode=mode)
            else:
                hvac.set_control(power=False, target=hvac.target_temp, fan=1)

            # 창문 권장: 솔루션 텍스트 전용, hvac 물리 시뮬레이션에 미반영
            window_rec = decide_window(
                pmv_val, out_temp, hvac.indoor_temp, heat_src, hvac.mode, people)

        # 1 sim-분마다 로그 기록
        if frame_idx % LOG_FRAMES == 0:
            h = sim_sec_int // 3600
            m = (sim_sec_int % 3600) // 60
            sol = _solution_text(
                sim_state.state, pmv_val,
                hvac.is_on, hvac.mode, hvac.fan_speed,
                hvac.target_temp, hvac.indoor_temp,
                window_rec=window_rec,
            )
            logs.append({
                "time":          f"{h:02d}:{m:02d}",
                "sim_sec":       sim_sec_int,
                "people":        people,
                "met":           met,
                "clo":           clo,
                "heat_source":   heat_src,
                "outdoor_temp":  out_temp,
                "outdoor_humid": out_humid,
                "indoor_temp":   round(hvac.indoor_temp,  2),
                "indoor_humid":  round(hvac.indoor_humid, 1),
                "pmv":           round(pmv_val, 2),
                "comfort":       engine.get_comfort_status(pmv_val),
                "state":         sim_state.state,
                "hvac_on":       hvac.is_on,
                "hvac_mode":     hvac.mode if hvac.is_on else "off",
                "fan_speed":     hvac.fan_speed,
                "target_temp":   hvac.target_temp,
                "window_rec":    ("open" if window_rec is True else
                                  "close" if window_rec is False else "keep"),
                "solution":      sol,
            })

        # 1 sim-시간마다 진행 상황 출력
        if frame_idx % PROGRESS_FRAMES == 0:
            h      = sim_sec_int // 3600
            status = (
                "냉방" if hvac.is_on and hvac.mode == "cool" else
                "난방" if hvac.is_on and hvac.mode == "heat" else
                "꺼짐"
            )
            sol = _solution_text(
                sim_state.state, pmv_val,
                hvac.is_on, hvac.mode, hvac.fan_speed,
                hvac.target_temp, hvac.indoor_temp,
                window_rec=window_rec,
            )
            print(
                f"  {h:02d}:00 | 실내 {hvac.indoor_temp:5.1f}°C"
                f" | PMV {pmv_val:+.2f}"
                f" | {status} Fan{hvac.fan_speed}"
                f" | 인원 {people}명"
                f" | [{sim_state.state}]"
                f"  >>  {sol}"
            )

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    df        = pd.DataFrame(logs)
    safe_name = name.replace(" ", "_").replace("/", "_")

    csv_path = output_dir / f"{safe_name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    png_path = output_dir / f"{safe_name}.png"
    try:
        _plot(df, name, png_path)
    except Exception as e:
        print(f"  [!] 그래프 생성 실패: {e}")
        png_path = None

    # ── 요약 통계 ─────────────────────────────────────────────────────────────
    total        = len(df)
    cool_min     = int((df["hvac_mode"] == "cool").sum())
    heat_min     = int((df["hvac_mode"] == "heat").sum())
    occupied     = df[df["people"] > 0]
    comfort_pct  = int((occupied["pmv"].abs() <= 0.5).sum() * 100 / len(occupied)) if len(occupied) > 0 else 0

    # 상태별 체류 시간
    state_counts = df["state"].value_counts()
    state_summary = "  ".join(
        f"{s}: {state_counts.get(s, 0)}분"
        for s in ["EMPTY", "ARRIVAL", "STEADY", "PRE_DEPARTURE"]
    )

    print(f"\n  -- 요약 --------------------------------------------------")
    print(f"  냉방 가동: {cool_min:4d}분  ({cool_min*100//total:2d}%)"
          f"  |  난방 가동: {heat_min:4d}분  ({heat_min*100//total:2d}%)")
    print(f"  PMV 범위: {df['pmv'].min():+.2f} ~ {df['pmv'].max():+.2f}"
          f"  |  쾌적 유지율: {comfort_pct}%")
    print(f"  실내온도 범위: {df['indoor_temp'].min():.1f}C"
          f" ~ {df['indoor_temp'].max():.1f}C")
    print(f"  상태 분포: {state_summary}")
    print(f"\n  -- 솔루션 샘플 (매 2시간) --------------------------------")
    for _, row in df.iloc[::120].iterrows():
        print(f"    {row['time']}  [{row['state']:14s}]  {row['solution']}")
    print(f"\n  CSV  -> {csv_path}")
    if png_path:
        print(f"  PNG  -> {png_path}")

    return df


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HVAC 시나리오 시뮬레이터 — 카메라·API 없이 24시간 제어 로직 고속 검증"
    )
    parser.add_argument(
        "--scenario", type=str, default=None, metavar="PATH",
        help="실행할 시나리오 JSON 경로 (생략 시 scenarios/*.json 전체 실행)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", metavar="DIR",
        help="결과 저장 폴더 (기본: results/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.scenario:
        scenario_files = [Path(args.scenario)]
    else:
        scenario_dir = Path("scenarios")
        if not scenario_dir.exists():
            print("오류: scenarios/ 폴더가 없습니다.")
            sys.exit(1)
        scenario_files = sorted(scenario_dir.glob("*.json"))
        if not scenario_files:
            print("오류: scenarios/ 폴더에 JSON 파일이 없습니다.")
            sys.exit(1)

    print(f"총 {len(scenario_files)}개 시나리오 실행 예정")

    for spath in scenario_files:
        with open(spath, encoding="utf-8") as f:
            scenario = json.load(f)
        run_scenario(scenario, output_dir)

    print(f"\n{'=' * 62}")
    print(f"  Done!  결과 위치: {output_dir.resolve()}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
