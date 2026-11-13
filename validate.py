"""실제 실행 vs 재생기 대조 검증.

실제 파이프라인(VLM+YOLO 포함)을 돌린 로그를 재생기에 다시 먹여,
제어·물리·에너지 체인이 동일한 결과를 내는지 확인한다.
같은 인식 결과를 입력하므로, 차이가 나면 재생기 쪽 문제다.
"""
import sys, pandas as pd, replay, importlib

real_csv = sys.argv[1]
d = pd.read_csv(real_csv)
occ = d[d.ai_pmv.notna()]

print("=" * 62)
print("  실제 실행 (VLM+YOLO 포함, 보드에서 20분 구동)")
print("=" * 62)
print(f"  로그 행수        : {len(d)}  (재실 {len(occ)})")
print(f"  영상 길이        : {d.video_time_sec.iloc[-1]:.0f} s")
print(f"  쾌적률           : {(occ.ai_pmv.abs()<=0.5).mean()*100:.1f}%")
print(f"  룰베이스 쾌적률  : {(d[d.rb_pmv.notna()].rb_pmv.abs()<=0.5).mean()*100:.1f}%")
print(f"  AI 에너지        : {d.ai_energy_wh.iloc[-1]:.1f} Wh")
print(f"  RB 에너지        : {d.rb_energy_wh.iloc[-1]:.1f} Wh")
print(f"  절감률           : {(1-d.ai_energy_wh.iloc[-1]/d.rb_energy_wh.iloc[-1])*100:.1f}%")
print(f"  평균 실내온도    : {d.ai_indoor_temp.mean():.2f} °C")
print(f"  PMV 범위         : {occ.ai_pmv.min():+.2f} ~ {occ.ai_pmv.max():+.2f}")

importlib.reload(replay)
r = replay.replay(real_csv, room_m2=30.0,
                  init_temp=float(d.ai_indoor_temp.iloc[0]),
                  init_humid=float(d.ai_indoor_humid.iloc[0]))
print()
print("=" * 62)
print("  재생기 (같은 인식 결과를 입력, 제어·물리·에너지만 재계산)")
print("=" * 62)
print(f"  쾌적률           : {r['comfort']:.1f}%")
print(f"  룰베이스 쾌적률  : {r['rb_comfort']:.1f}%")
print(f"  AI 에너지        : {r['ai_wh']:.1f} Wh")
print(f"  RB 에너지        : {r['rb_wh']:.1f} Wh")
print(f"  절감률           : {r['savings']:.1f}%")
print(f"  평균 실내온도    : {r['ai_temp']:.2f} °C")

print()
print("=" * 62)
print("  대조")
print("=" * 62)
rows = [
    ("쾌적률(%)",   (occ.ai_pmv.abs()<=0.5).mean()*100, r["comfort"]),
    ("RB쾌적(%)",   (d[d.rb_pmv.notna()].rb_pmv.abs()<=0.5).mean()*100, r["rb_comfort"]),
    ("AI에너지(Wh)", d.ai_energy_wh.iloc[-1], r["ai_wh"]),
    ("RB에너지(Wh)", d.rb_energy_wh.iloc[-1], r["rb_wh"]),
    ("절감률(%)",   (1-d.ai_energy_wh.iloc[-1]/d.rb_energy_wh.iloc[-1])*100, r["savings"]),
    ("실내온도(°C)", d.ai_indoor_temp.mean(), r["ai_temp"]),
]
print(f"  {'항목':<14}{'실제':>10}{'재생':>10}{'차이':>10}")
print("  " + "-" * 44)
for name, a, b in rows:
    diff = b - a
    rel = abs(diff) / abs(a) * 100 if a else 0
    mark = "OK" if rel < 5 else "★차이큼"
    print(f"  {name:<14}{a:>10.1f}{b:>10.1f}{diff:>+10.1f}  {mark}")
