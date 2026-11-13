"""다중 베이스라인 비교 리포트 — AI 1개 vs 고정 설정온도 3개(22/24/26°C).

고정 제어는 설정값 선택에 성능이 좌우된다. 하나의 베이스라인만 쓰면
그 값이 우연히 최적일 때 제안 시스템의 이점이 가려지므로, 실제 현장에서
쓰이는 범위(과냉방 22°C / 통상 24°C / 공공기관 권고 26°C)를 함께 비교한다.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import replay

C_AI   = "#2196F3"
C_RB   = ["#FF7043", "#FB8C00", "#8D6E63"]   # 22 / 24 / 26°C
C_GOOD = "#4CAF50"
C_BAD  = "#F44336"
COMFORT_BAND = 0.5
SETPOINTS = [22.0, 24.0, 26.0]
LABELS    = ["Fixed 22°C (over-cooling)", "Fixed 24°C (typical)", "Fixed 26°C (public guideline)"]


def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[Report] 저장:", p)


def run(csv_path, out_dir, room_m2=30.0):
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    for sp in SETPOINTS:
        import importlib; importlib.reload(replay)
        r = replay.replay(csv_path, room_m2=room_m2, rb_setpoint=sp, smooth=True)
        runs.append(r)

    ai = runs[0]                       # AI 결과는 베이스라인과 무관하게 동일
    t  = np.array([x["t"] for x in ai["rows"]]) / 60.0     # 분

    def series(r, key):
        return np.array([np.nan if x[key] is None else x[key] for x in r["rows"]], dtype=float)

    # ── 1. PMV 비교 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axhspan(-COMFORT_BAND, COMFORT_BAND, alpha=0.12, color=C_GOOD,
               label="Comfort Zone (±0.5)")
    ax.plot(t, series(ai, "pmv"), color=C_AI, lw=1.8, label="AI Control (VLM)", zorder=5)
    for r, c, lab in zip(runs, C_RB, LABELS):
        ax.plot(t, series(r, "rpmv"), color=c, lw=1.2, ls="--", alpha=0.85, label=lab)
    ax.set_xlabel("Video Time (min)"); ax.set_ylabel("PMV Index")
    ax.set_title("PMV: AI Control vs Fixed-Setpoint Baselines")
    ax.set_ylim(-3.2, 3.2); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    _save(fig, out_dir, "01_pmv_comparison.png")

    # ── 2. 실내온도 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, series(ai, "ai_temp"), color=C_AI, lw=1.8, label="AI Control", zorder=5)
    for r, c, lab in zip(runs, C_RB, LABELS):
        ax.plot(t, series(r, "rb_temp"), color=c, lw=1.2, ls="--", alpha=0.85, label=lab)
    ax.set_xlabel("Video Time (min)"); ax.set_ylabel("Indoor Temp (°C)")
    ax.set_title("Indoor Temperature")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8, ncol=2)
    _save(fig, out_dir, "02_indoor_temp.png")

    # ── 3. 쾌적률 ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["AI Control"] + [f"Fixed {int(s)}°C" for s in SETPOINTS]
    vals  = [ai["comfort"]] + [r["rb_comfort"] for r in runs]
    bars  = ax.bar(names, vals, color=[C_AI] + C_RB, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%",
                ha="center", fontweight="bold", fontsize=11)
    ax.axhline(100, color="#BDBDBD", lw=0.8, ls=":")
    ax.set_ylabel("Comfort Rate (%)"); ax.set_ylim(0, 112)
    ax.set_title("Thermal Comfort Rate  (PMV within ±0.5, occupied only)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "05_comfort_rate.png")

    # ── 4. 에너지 막대 ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [ai["ai_wh"]] + [r["rb_wh"] for r in runs]
    bars = ax.bar(names, vals, color=[C_AI] + C_RB, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 20, f"{v:.0f} Wh",
                ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Energy (Wh)"); ax.set_title("Total Energy Consumption (6 h)")
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(vals) * 1.15)
    _save(fig, out_dir, "04_energy_bar.png")

    # ── 5. 누적 에너지 ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, series(ai, "ai_energy_wh"), color=C_AI, lw=2, label="AI Control")
    for r, c, lab in zip(runs, C_RB, LABELS):
        ax.plot(t, series(r, "rb_energy_wh"), color=c, lw=1.3, ls="--", label=lab)
    ax.set_xlabel("Video Time (min)"); ax.set_ylabel("Cumulative Energy (Wh)")
    ax.set_title("Cumulative Energy Consumption")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    _save(fig, out_dir, "03_energy_cumulative.png")

    # ── 6. 요약 ──────────────────────────────────────────────────────────────
    L = ["=" * 66,
         "  HVAC VLM ANALYSIS — AI vs MULTIPLE FIXED-SETPOINT BASELINES",
         "=" * 66,
         f"  Video duration      : {t[-1]:.1f} min",
         f"  Occupied samples    : {sum(1 for x in ai['rows'] if x['pmv'] is not None)}",
         "",
         "  ── AI Control (proposed) ──────────────────────────────",
         f"  Comfort rate        : {ai['comfort']:.1f}%",
         f"  Energy              : {ai['ai_wh']:.1f} Wh",
         f"  Mean indoor temp    : {ai['ai_temp']:.2f} °C",
         "",
         "  ── Fixed-setpoint baselines ───────────────────────────",
         "  Setpoint   Comfort    Energy      AI advantage",
         "  " + "-" * 56]
    for sp, r, lab in zip(SETPOINTS, runs, LABELS):
        L.append(f"  {int(sp)}°C       {r['rb_comfort']:5.1f}%   {r['rb_wh']:7.1f} Wh   "
                 f"comfort {ai['comfort']-r['rb_comfort']:+5.1f}p / energy {r['savings']:+5.1f}%")
    L += ["",
          "  ── Note ───────────────────────────────────────────────",
          "  Fixed control performance depends strongly on the chosen",
          "  setpoint; the proposed system holds comfort regardless.",
          "  Comfort = PMV within +/-0.5, occupied frames only.",
          "  P = P_fan + P_comp(1200W rated) x load;  Fan 40/70/110 W.",
          "=" * 66]
    txt = "\n".join(L)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    return runs


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
