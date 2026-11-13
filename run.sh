#!/bin/bash
# Jetson Orin Nano Super (JetPack 6.x, user: jetson) 실행 스크립트
# - 가상환경 자동 탐색 (vlm-env / .venv)
# - DISPLAY 자동 감지 (재부팅마다 바뀜 — X11 소켓으로 찾기)
# - DISPLAY 없으면 headless 모드로 자동 전환 (main.py가 감지)
#
# ※ 구 보드(hanul, JetPack 5.1.2)는 libgomp/libGLdispatch LD_PRELOAD가
#   필요했으나, 새 보드(JetPack 6.x)에서는 불필요.
cd "$(dirname "$0")"

# 가상환경 활성화
for VENV in ./vlm-env "$HOME/vlm-env" ./.venv; do
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
        break
    fi
done

# DISPLAY 자동 감지 — 현재 사용자가 실제로 열 수 있는 X 디스플레이만 선택
# (다중 사용자 보드: 물리 모니터(:1)는 다른 계정 소유, VNC 세션(:3 등)은 내 것)
if [ -z "$DISPLAY" ]; then
    for SOCK in $(ls /tmp/.X11-unix/ 2>/dev/null); do
        D="${SOCK/X/:}"
        if DISPLAY="$D" timeout 2 xset q >/dev/null 2>&1; then
            export DISPLAY="$D"; break
        fi
    done
    [ -z "$DISPLAY" ] && echo "[run.sh] 열 수 있는 X 디스플레이 없음 — headless 모드"
fi

# Jetson 통합 메모리: CUDA VMM이 OOM 유발 → cudaMalloc 강제
export GGML_CUDA_NO_VMM=1
export PYTHONUNBUFFERED=1

exec python3 main.py "$@"
