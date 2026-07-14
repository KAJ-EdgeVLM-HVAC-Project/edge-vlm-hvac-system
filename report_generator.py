"""
[영상 분석 결과 리포트 생성기]
video_mode() 완료 후 호출 — CSV 데이터로 논문용 그래프와 요약 통계를 자동 생성합니다.

생성 파일:
  01_pmv_comparison.png      : AI vs 룰베이스 PMV 시계열 비교
  02_indoor_temp.png         : 실내온도 비교 (AI / 룰베이스 / 목표온도)
  03_energy_cumulative.png   : 누적 에너지 소비량 비교
  04_energy_bar.png          : 총 에너지 절감 막대 그래프
  05_comfort_rate.png        : 쾌적 구간(PMV ±0.5) 유지율 파이차트
  06_activity_distribution.png : VLM 활동 분류 분포
  summary.txt                : 수치 요약 (논문 표에 바로 사용 가능)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

rcParams["font.family"]      = "DejaVu Sans"
rcParams["axes.unicode_minus"] = False

# ── 색상 팔레트 (논문 친화적) ────────────────────────────────────────────────
C_AI   = "#2196F3"   # 파랑 — AI 제어
C_RB   = "#FF7043"   # 주황 — 룰베이스
C_GOOD = "#4CAF50"   # 초록 — 쾌적
C_BAD  = "#F44336"   # 빨강 — 불쾌적
C_GRAY = "#9E9E9E"

COMFORT_BAND = 0.5   # PMV 쾌적 기준 ±0.5  (ASHRAE 55 / ISO 7730)


def generate(csv_path: str, output_dir: str) -> dict:
    """
    CSV 로그를 읽어 모든 그래프와 summary.txt를 output_dir에 저장.
    반환값: 핵심 수치 딕셔너리 (summary에도 저장됨)
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty or "video_time_sec" not in df.columns:
        print("[Report] 유효한 video_mode CSV 아님 — 리포트 건너뜀")
        return {}

    df = df.sort_values("video_time_sec").reset_index(drop=True)
    t  = df["video_time_sec"].values   # x축 공통

    stats = {}

    # ── 1. PMV 비교 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, df["ai_pmv"],  color=C_AI, lw=1.8, label="AI Control (VLM)")
    ax.plot(t, df["rb_pmv"],  color=C_RB, lw=1.4, ls="--", label="Rule-based (Fixed 24°C)")
    ax.axhspan(-COMFORT_BAND, COMFORT_BAND, alpha=0.12, color=C_GOOD, label="Comfort Zone (±0.5)")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Video Time (s)")
    ax.set_ylabel("PMV Index")
    ax.set_title("PMV Comparison: AI Control vs Rule-based Baseline")
    ax.set_ylim(-3.2, 3.2)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, output_dir, "01_pmv_comparison.png")

    # 쾌적율 계산
    ai_comfort = (df["ai_pmv"].abs() <= COMFORT_BAND).mean() * 100
    rb_comfort = (df["rb_pmv"].abs() <= COMFORT_BAND).mean() * 100
    stats["ai_comfort_rate_pct"]  = round(ai_comfort, 1)
    stats["rb_comfort_rate_pct"]  = round(rb_comfort, 1)
    stats["comfort_improvement_ppt"] = round(ai_comfort - rb_comfort, 1)

    # ── 2. 실내온도 비교 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, df["ai_indoor_temp"], color=C_AI,  lw=1.8, label="AI Indoor Temp")
    ax.plot(t, df["rb_indoor_temp"], color=C_RB,  lw=1.4, ls="--", label="Rule-based Indoor Temp")
    if "outdoor_temp" in df.columns:
        ax.plot(t, df["outdoor_temp"],   color=C_GRAY, lw=1.0, ls=":", label="Outdoor Temp")
    ax.axhline(24.0, color="black", lw=0.8, ls="-.", alpha=0.5, label="Setpoint 24°C")
    ax.set_xlabel("Video Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Indoor Temperature: AI vs Rule-based")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, output_dir, "02_indoor_temp.png")

    stats["ai_mean_indoor_temp"] = round(df["ai_indoor_temp"].mean(), 2)
    stats["rb_mean_indoor_temp"] = round(df["rb_indoor_temp"].mean(), 2)

    # ── 3. 누적 에너지 소비 ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, df["ai_energy_wh"],  color=C_AI, lw=2.0, label="AI Control")
    ax.plot(t, df["rb_energy_wh"],  color=C_RB, lw=1.6, ls="--", label="Rule-based")
    ax.fill_between(t, df["ai_energy_wh"], df["rb_energy_wh"],
                    where=df["rb_energy_wh"] >= df["ai_energy_wh"],
                    alpha=0.15, color=C_GOOD, label="Energy Saved")
    ax.set_xlabel("Video Time (s)")
    ax.set_ylabel("Cumulative Energy (Wh)")
    ax.set_title("Cumulative Energy Consumption")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _savefig(fig, output_dir, "03_energy_cumulative.png")

    ai_total_wh = df["ai_energy_wh"].iloc[-1]
    rb_total_wh = df["rb_energy_wh"].iloc[-1]
    saved_wh    = rb_total_wh - ai_total_wh
    save_pct    = (saved_wh / rb_total_wh * 100) if rb_total_wh > 0 else 0.0
    stats["ai_total_energy_wh"]  = round(ai_total_wh, 2)
    stats["rb_total_energy_wh"]  = round(rb_total_wh, 2)
    stats["energy_saved_wh"]     = round(saved_wh, 2)
    stats["energy_saving_pct"]   = round(save_pct, 1)

    # ── 4. 에너지 절감 막대 그래프 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Rule-based\n(Fixed 24°C)", "AI Control\n(VLM-based)"],
                  [rb_total_wh, ai_total_wh],
                  color=[C_RB, C_AI], width=0.45, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [rb_total_wh, ai_total_wh]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f} Wh", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.annotate(f"Saved: {saved_wh:.1f} Wh\n({save_pct:.1f}%)",
                xy=(1, ai_total_wh), xytext=(1.35, (rb_total_wh + ai_total_wh) / 2),
                fontsize=10, color=C_GOOD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GOOD))
    ax.set_ylabel("Total Energy (Wh)")
    ax.set_title("Energy Consumption Comparison\n(Simulated, based on compressor-load power model)")
    ax.set_ylim(0, rb_total_wh * 1.25)
    ax.grid(axis="y", alpha=0.3)
    # 주석: 전력 모델 근거
    ax.text(0.01, 0.02,
            "Power model: P = P_fan + P_comp(1200W) x load\n"
            "load = heat-gain / cooling-capacity (setpoint-aware)\n"
            "Rule-based baseline: fixed 24C + Fan2 when occupied",
            transform=ax.transAxes, fontsize=7, color=C_GRAY, va="bottom")
    _savefig(fig, output_dir, "04_energy_bar.png")

    # ── 5. 쾌적율 파이차트 ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, label, rate, color in [
        (axes[0], "AI Control", ai_comfort, C_AI),
        (axes[1], "Rule-based", rb_comfort, C_RB),
    ]:
        ax.pie(
            [rate, 100 - rate],
            labels=[f"Comfort\n{rate:.1f}%", f"Discomfort\n{100-rate:.1f}%"],
            colors=[C_GOOD, "#FFCCBC"],
            startangle=90, autopct=None,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax.set_title(label, fontsize=12, fontweight="bold", color=color)
    fig.suptitle("Thermal Comfort Rate  (PMV within ±0.5, ASHRAE 55)", fontsize=12)
    _savefig(fig, output_dir, "05_comfort_rate.png")

    # ── 6. VLM 활동 분포 ─────────────────────────────────────────────────────
    if "vlm_activity" in df.columns:
        acts  = df["vlm_activity"].dropna().value_counts()
        total = acts.sum()
        colors = plt.cm.Set2(np.linspace(0, 1, len(acts)))
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(acts.index, acts.values, color=colors)
        for bar, val in zip(bars, acts.values):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{val}  ({val/total*100:.1f}%)", va="center", fontsize=9)
        ax.set_xlabel("Frame Count")
        ax.set_title("VLM Activity Classification Distribution")
        ax.grid(axis="x", alpha=0.3)
        _savefig(fig, output_dir, "06_activity_distribution.png")

    # ── 7. Summary.txt 저장 ───────────────────────────────────────────────────
    video_dur = t[-1] if len(t) > 0 else 0
    n_vlm     = len(df)
    summary_lines = [
        "=" * 60,
        "  HVAC VLM ANALYSIS SUMMARY",
        "=" * 60,
        f"  Video duration          : {video_dur:.1f} s  ({video_dur/60:.1f} min)",
        f"  VLM analysis frames     : {n_vlm}",
        "",
        "  ── Energy (Simulated) ─────────────────────────",
        f"  AI Control total        : {stats['ai_total_energy_wh']:.2f} Wh",
        f"  Rule-based total        : {stats['rb_total_energy_wh']:.2f} Wh",
        f"  Energy saved            : {stats['energy_saved_wh']:.2f} Wh  ({stats['energy_saving_pct']:.1f}%)",
        "",
        "  ── Thermal Comfort (PMV ±0.5) ─────────────────",
        f"  AI Control comfort rate : {stats['ai_comfort_rate_pct']:.1f}%",
        f"  Rule-based comfort rate : {stats['rb_comfort_rate_pct']:.1f}%",
        f"  Improvement             : {stats['comfort_improvement_ppt']:+.1f} ppt",
        "",
        "  ── Temperature ────────────────────────────────",
        f"  AI mean indoor temp     : {stats['ai_mean_indoor_temp']:.2f} °C",
        f"  Rule-based mean temp    : {stats['rb_mean_indoor_temp']:.2f} °C",
        "",
        "  ── Energy Model Note ──────────────────────────",
        "  P = P_fan + P_comp(1200W rated) x load",
        "  load = heat-gain / cooling-capacity (setpoint-aware)",
        "  Fan: 40/70/110 W (speed 1/2/3)",
        "  Baseline: fixed 24C + Fan2 during occupancy",
        "  Figures saved in: " + output_dir,
        "=" * 60,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    return stats


def _savefig(fig, output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Report] 저장: {path}")
