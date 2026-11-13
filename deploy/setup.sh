#!/bin/bash
# 새 리눅스 머신 초기 설정 — 저장소 clone 후 이 스크립트 한 번만 실행.
#
#   Jetson (JetPack)  : llama.cpp CUDA 빌드 + Qwen3-VL GGUF 다운로드 (INT4 온디바이스)
#   일반 Linux + GPU  : ultralytics/transformers 경로 (torch 필요, 별도 설치)
#
# TensorRT 엔진은 보드마다 달라서 여기서 만들지 않는다.
# 첫 실행 때 yolo_detector.py 가 이 보드용으로 자동 생성한다.
set -e
PROJECT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$PROJECT"

is_jetson() { [ -f /etc/nv_tegra_release ]; }

echo "══ Smart HVAC 설치 ══"
echo "경로: $PROJECT"
is_jetson && echo "보드: Jetson ($(tr -d '\0' </proc/device-tree/model 2>/dev/null))" \
          || echo "보드: 일반 Linux ($(uname -m))"

# ── 1. Python 가상환경 ───────────────────────────────────────────────────────
# Jetson은 cv2(GTK 빌드)·tensorrt가 시스템 패키지로만 제공되므로 공유해야 한다.
if [ ! -d .venv ]; then
    if is_jetson; then
        python3 -m venv --system-site-packages .venv
    else
        python3 -m venv .venv
    fi
    echo "✅ .venv 생성"
fi
./.venv/bin/pip install -q --upgrade pip

if is_jetson; then
    ./.venv/bin/pip install -q -r requirements_jetson.txt
    # TensorRT 엔진을 torch 없이 직접 구동하기 위해 필요
    CUDA_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' \
                 || ls -d /usr/local/cuda-* 2>/dev/null | tail -1 | grep -oP 'cuda-\K[0-9]+')
    ./.venv/bin/pip install -q "cuda-python>=${CUDA_MAJOR:-12},<$(( ${CUDA_MAJOR:-12} + 1 ))"
    # 폴백 경로용. --no-deps: opencv-python(Qt)·numpy 2.x가 시스템 cv2/tensorrt를
    # 가리는 것을 막는다 (requirements_jetson.txt 주석 참고).
    ./.venv/bin/pip install -q --no-deps ultralytics
    # 시스템 패키지를 가리는 사본이 생겼는지 확인 — 생겼다면 제거해야 창이 뜬다
    ./.venv/bin/python - <<'PY'
import cv2, numpy, sys
bad = [m.__name__ for m in (cv2, numpy) if "/.venv/" in (m.__file__ or "")]
if bad or numpy.__version__.startswith("2"):
    sys.exit(f"❌ {bad or 'numpy 2.x'} 가 시스템 패키지를 가림 — "
             f"'.venv/bin/pip uninstall -y opencv-python numpy' 후 다시 실행하세요")
print(f"✅ cv2 {cv2.__version__} / numpy {numpy.__version__} (시스템 빌드 사용)")
PY
else
    ./.venv/bin/pip install -q -r requirements.txt
    echo "ℹ️  GPU 추론을 쓰려면 torch를 따로 설치하세요: https://pytorch.org"
fi
echo "✅ 의존성 설치"

# ── 2. 환경변수 파일 ─────────────────────────────────────────────────────────
[ -f .env ] || { cp .env.example .env; echo "✅ .env 생성 (기상청 API 키는 선택)"; }

# ── 3. Jetson 전용: llama.cpp + VLM 모델 ─────────────────────────────────────
if is_jetson; then
    LCPP="$HOME/llama.cpp"
    if [ ! -x "$LCPP/build/bin/llama-server" ]; then
        echo "── llama.cpp CUDA 빌드 (10~20분) ──"
        [ -d "$LCPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LCPP"
        CUDA_DIR=$(ls -d /usr/local/cuda-*/ 2>/dev/null | sort -V | tail -1)
        CUDA_DIR=${CUDA_DIR:-/usr/local/cuda/}
        # Orin=87, Xavier=72. 실패 시 둘 다 넣어 빌드.
        ARCH=$(./.venv/bin/python -c "
import os;m=open('/proc/device-tree/model','rb').read().decode('utf8','ignore')
print(87 if 'Orin' in m else 72 if 'Xavier' in m else '87;72')" 2>/dev/null || echo "87;72")
        ./.venv/bin/pip install -q cmake ninja
        PATH="$PROJECT/.venv/bin:${CUDA_DIR}bin:$PATH" \
        ./.venv/bin/cmake -B "$LCPP/build" -S "$LCPP" -G Ninja \
            -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
            -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCUDAToolkit_ROOT="${CUDA_DIR%/}" \
            -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
        PATH="$PROJECT/.venv/bin:${CUDA_DIR}bin:$PATH" \
        ./.venv/bin/cmake --build "$LCPP/build" --config Release \
            -j"$(nproc)" --target llama-server llama-mtmd-cli
        echo "✅ llama.cpp 빌드 완료"
    else
        echo "✅ llama.cpp 이미 빌드됨"
    fi

    MODEL_DIR="$LCPP/models/Qwen3-VL-2B"
    BASE=https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main
    mkdir -p "$MODEL_DIR"
    for f in Qwen3VL-2B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-F16.gguf; do
        if [ ! -s "$MODEL_DIR/$f" ]; then
            echo "── 모델 다운로드: $f ──"
            curl -L --retry 3 --progress-bar -o "$MODEL_DIR/$f" "$BASE/$f"
        fi
    done
    echo "✅ VLM 모델 준비 (약 1.9GB)"
fi

# ── 4. 검증 + 런처 ───────────────────────────────────────────────────────────
./.venv/bin/python -m pytest tests/ -q
bash "$PROJECT/deploy/install-launcher.sh"

cat <<EOF

══ 설치 완료 ══
  실행:  ./run.sh                          (시작화면 → 카메라/영상 선택)
         ./run.sh --video 영상.mp4          (영상 분석 + 그래프 자동 생성)
         HVAC_HEADLESS=1 ./run.sh          (화면 없이 콘솔+CSV)
  또는 바탕화면 'Smart HVAC' 아이콘 더블클릭

  ※ 첫 실행 시 YOLO TensorRT 엔진을 이 보드용으로 생성합니다 (수 분, 1회만)
EOF
