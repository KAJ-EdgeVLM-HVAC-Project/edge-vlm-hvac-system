#!/usr/bin/env python3
"""
[엣지 추론 성능 벤치마크]  — 논문 4.4절(엣지 추론 성능)용

Jetson 보드에서 실제 VLM 추론 경로(vlm_processor)를 그대로 사용해
다음을 측정한다:
  · 백엔드(lcpp/cuda/mps/cpu) 및 모델 파일 크기(GGUF + mmproj)
  · 콜드 스타트(모델 1회 로드) 시간
  · 추론 1회 지연시간 (warm-up 후 N회: 평균/중앙/최소/최대/표준편차)
  · 시스템 메모리 사용량 + GPU 사용률 + 소비 전력 (tegrastats, Jetson 한정)

사용법 (보드에서):
    cd ~/edge-vlm-hvac-system
    python bench_edge.py                      # 합성 720p 프레임으로 측정
    python bench_edge.py --image frame.jpg     # 실제 이미지로 측정
    python bench_edge.py --video classroom.mp4 # 영상 첫 프레임으로 측정
    python bench_edge.py --runs 20             # 반복 횟수 지정

결과는 results/edge_benchmark_<시각>.json 에 저장되며, 논문 표에 바로 쓸 수 있다.
Mac에서도 동작하지만(백엔드=mps, 전력측정 생략) 정식 수치는 보드에서 뽑을 것.
"""
import os
import re
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


# ── tegrastats (Jetson 전력/메모리/GPU) ───────────────────────────────────────
class TegraStats:
    """tegrastats를 백그라운드로 돌려 RAM·GPU%·전력을 수집/평균."""
    RAM_RE   = re.compile(r"RAM (\d+)/(\d+)MB")
    GR3D_RE  = re.compile(r"GR3D_FREQ (\d+)%")
    POWER_RE = re.compile(r"VDD_IN (\d+)mW")           # 보드 입력 총전력
    POWER_ANY_RE = re.compile(r"([A-Z0-9_]+) (\d+)mW") # 폴백: 모든 전력 레일

    def __init__(self):
        self.proc = None
        self.samples = []   # (ram_used, ram_total, gr3d, power_mw)

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            self.proc = None   # Jetson이 아니면 tegrastats 없음
        return self.proc is not None

    def _drain(self):
        if not self.proc or not self.proc.stdout:
            return
        # non-blocking이 아니므로 종료 후 일괄 파싱
        for line in self.proc.stdout:
            ram = self.RAM_RE.search(line)
            g   = self.GR3D_RE.search(line)
            p   = self.POWER_RE.search(line)
            if not p:
                rails = self.POWER_ANY_RE.findall(line)
                pw = sum(int(v) for _, v in rails) if rails else None
            else:
                pw = int(p.group(1))
            self.samples.append((
                int(ram.group(1)) if ram else None,
                int(ram.group(2)) if ram else None,
                int(g.group(1)) if g else None,
                pw,
            ))

    def stop(self):
        if not self.proc:
            return None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        self._drain()
        if not self.samples:
            return None
        def avg(idx):
            vals = [s[idx] for s in self.samples if s[idx] is not None]
            return round(sum(vals) / len(vals), 1) if vals else None
        def peak(idx):
            vals = [s[idx] for s in self.samples if s[idx] is not None]
            return max(vals) if vals else None
        return {
            "ram_used_mb_avg":  avg(0),
            "ram_used_mb_peak": peak(0),
            "ram_total_mb":     self.samples[-1][1],
            "gpu_util_pct_avg": avg(2),
            "gpu_util_pct_peak": peak(2),
            "power_mw_avg":     avg(3),
            "power_mw_peak":    peak(3),
            "n_samples":        len(self.samples),
        }


def make_test_frame(w=1280, h=720):
    """합성 720p 프레임 (사람 비슷한 사각형 몇 개) — 지연시간 측정용."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    if cv2 is not None:
        for i, x in enumerate(range(120, w - 200, 220)):
            cv2.rectangle(img, (x, 250), (x + 130, 600), (90, 90, 120), -1)
            cv2.circle(img, (x + 65, 210), 55, (160, 140, 130), -1)
        cv2.putText(img, "bench", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 0), 3)
    return img


def load_frame(args):
    if args.image and cv2 is not None:
        f = cv2.imread(args.image)
        if f is not None:
            return f, f"image:{os.path.basename(args.image)}"
    if args.video and cv2 is not None:
        cap = cv2.VideoCapture(args.video)
        ok, f = cap.read(); cap.release()
        if ok:
            return f, f"video:{os.path.basename(args.video)}"
    return make_test_frame(), "synthetic-720p"


def fmt(v, unit=""):
    return f"{v}{unit}" if v is not None else "N/A"


def main():
    ap = argparse.ArgumentParser(description="엣지 추론 성능 벤치마크 (논문 4.4)")
    ap.add_argument("--runs", type=int, default=10, help="측정 반복 횟수 (기본 10)")
    ap.add_argument("--warmup", type=int, default=2, help="워밍업 횟수 (기본 2)")
    ap.add_argument("--image", type=str, default=None, help="테스트 이미지 경로")
    ap.add_argument("--video", type=str, default=None, help="영상(첫 프레임 사용) 경로")
    args = ap.parse_args()

    print("=" * 60)
    print("  엣지 추론 성능 벤치마크 (논문 4.4절)")
    print("=" * 60)

    # ── 1. 모델 로드 (콜드 스타트) ───────────────────────────────────────────
    from vlm_processor import VLMProcessor

    t0 = time.time()
    vlm = VLMProcessor()
    load_sec = time.time() - t0
    backend = vlm.device
    print(f"\n  백엔드        : {backend}")
    print(f"  콜드 스타트   : {load_sec:.2f} s (모델 로드/서버 기동)")

    # ── 2. 모델 파일 크기 ────────────────────────────────────────────────────
    model_info = {}
    try:
        gguf, mmproj = VLMProcessor._find_lcpp_model()
        if gguf and os.path.exists(gguf):
            sz = os.path.getsize(gguf) / 1e6
            model_info["gguf"] = {"path": gguf, "mb": round(sz, 1)}
            print(f"  모델(GGUF)    : {os.path.basename(gguf)}  {sz:.0f} MB")
        if mmproj and os.path.exists(mmproj):
            sz = os.path.getsize(mmproj) / 1e6
            model_info["mmproj"] = {"path": mmproj, "mb": round(sz, 1)}
            print(f"  비전(mmproj)  : {os.path.basename(mmproj)}  {sz:.0f} MB")
    except Exception as e:
        print(f"  (모델 파일 탐색 생략: {e})")

    # ── 3. 추론 지연시간 측정 (+ tegrastats 동시 수집) ──────────────────────
    frame, frame_src = load_frame(args)
    print(f"  입력 프레임   : {frame_src}  {frame.shape[1]}x{frame.shape[0]}")

    print(f"\n  워밍업 {args.warmup}회...")
    for _ in range(args.warmup):
        vlm.analyze_frame(frame)

    teg = TegraStats()
    teg_on = teg.start()
    print(f"  측정 {args.runs}회 (tegrastats {'ON' if teg_on else '없음(비-Jetson)'}) ...")

    lat = []
    ok = 0
    for i in range(args.runs):
        s = time.time()
        r = vlm.analyze_frame(frame)
        dt = time.time() - s
        lat.append(dt)
        if r is not None:
            ok += 1
        print(f"    [{i+1:2d}/{args.runs}] {dt*1000:7.1f} ms   "
              f"{'clo=%.2f met=%.2f' % (r.get('clo',0), r.get('met',0)) if r else 'FAIL'}")

    power = teg.stop()

    # ── 4. 통계 ──────────────────────────────────────────────────────────────
    import statistics as st
    stats = {
        "mean_ms":   round(st.mean(lat) * 1000, 1),
        "median_ms": round(st.median(lat) * 1000, 1),
        "min_ms":    round(min(lat) * 1000, 1),
        "max_ms":    round(max(lat) * 1000, 1),
        "std_ms":    round(st.pstdev(lat) * 1000, 1),
        "fps":       round(1.0 / st.mean(lat), 3),
        "success":   f"{ok}/{args.runs}",
    }

    print("\n" + "=" * 60)
    print("  결과 요약 (논문 표 4.x)")
    print("=" * 60)
    print(f"  백엔드            : {backend}")
    print(f"  콜드 스타트       : {load_sec:.2f} s")
    print(f"  추론 평균         : {stats['mean_ms']} ms  (중앙 {stats['median_ms']} ms)")
    print(f"  추론 최소~최대    : {stats['min_ms']} ~ {stats['max_ms']} ms (σ={stats['std_ms']})")
    print(f"  처리율            : {stats['fps']} fps")
    print(f"  추론 성공         : {stats['success']}")
    if power:
        print(f"  시스템 메모리     : 평균 {fmt(power['ram_used_mb_avg'],' MB')} / "
              f"{fmt(power['ram_total_mb'],' MB')} (피크 {fmt(power['ram_used_mb_peak'],' MB')})")
        print(f"  GPU 사용률        : 평균 {fmt(power['gpu_util_pct_avg'],'%')} "
              f"(피크 {fmt(power['gpu_util_pct_peak'],'%')})")
        print(f"  소비 전력         : 평균 {fmt(power['power_mw_avg'],' mW')} "
              f"(피크 {fmt(power['power_mw_peak'],' mW')})")
    else:
        print("  전력/메모리       : (tegrastats 없음 — Jetson에서 실행 시 측정됨)")
    print("=" * 60)

    # ── 5. JSON 저장 ─────────────────────────────────────────────────────────
    out = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "cold_start_sec": round(load_sec, 2),
        "model": model_info,
        "frame_source": frame_src,
        "latency": stats,
        "tegrastats": power,
        "runs": args.runs,
    }
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results",
                        f"edge_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  저장됨: {path}")
    print("  → 이 수치를 논문 4.4절 '엣지 추론 성능' 표에 사용하세요.")


if __name__ == "__main__":
    main()
