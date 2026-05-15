#!/bin/bash
cd "$(dirname "$0")"
source /home/hanul/vlm-env/bin/activate

# DISPLAY 자동 감지 (재부팅마다 바뀜 — X11 소켓으로 찾기)
if [ -z "$DISPLAY" ]; then
    DISPLAY=$(ls /tmp/.X11-unix/ 2>/dev/null | head -1 | sed 's/X/:/')
    export DISPLAY
fi

XAUTHORITY=/run/user/1000/gdm/Xauthority \
  LD_PRELOAD="/home/hanul/vlm-env/lib/python3.8/site-packages/torch.libs/libgomp-804f19d4.so.1.0.0:/lib/aarch64-linux-gnu/libGLdispatch.so.0" \
  PYTHONUNBUFFERED=1 \
  python main.py "$@"
