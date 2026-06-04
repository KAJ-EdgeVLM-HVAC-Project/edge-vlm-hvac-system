# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Edge-deployed VLM-based intelligent HVAC control system. A camera analyzes occupants' clothing, posture, and activity via a Vision Language Model (Qwen2-VL-2B), computes PMV (ISO 7730:2005), and controls a simulated HVAC unit accordingly. Designed for Jetson Orin Nano Super deployment with Mac development support.

## Running the System

**Mac (development):**
```bash
source .venv/bin/activate
python main.py                  # 30s VLM interval (default)
python main.py --interval 10    # 10s interval (M-series Mac)
```

**Jetson Orin Nano Super (172.20.10.11, user: jetson):**
```bash
cd ~/edge-vlm-hvac-system
./run.sh                        # auto-detects DISPLAY, sets GGML_CUDA_NO_VMM=1
# or manually:
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority PYTHONUNBUFFERED=1 GGML_CUDA_NO_VMM=1 python3 main.py
```

**Environment setup (.env):**
```
WEATHER_API_KEY=...
AIR_QUALITY_API_KEY=...   # from data.go.kr
AIR_QUALITY_STATION=장림동
```

## Architecture

### Data Flow
```
Camera frame
  → YOLODetector      : people count, bounding boxes
  → VLMProcessor      : activity, clo, met, room_size, heat_source (via Qwen2-VL)
  → ThermalEngine     : PMV/PPD (ISO 7730:2005)
  → StateManager      : system state (EMPTY/ARRIVAL/STEADY/PRE_DEPARTURE/LUNCH_BREAK)
  → decide_control()  : power, target_temp, fan_speed, mode
  → HVACSimulator     : simulates indoor_temp, indoor_humid per frame
  → EnergyMonitor     : cumulative Wh, baseline comparison, comfort rate
  → Dashboard + UserDisplay : two OpenCV windows + VLM Context window
  → CSV log (hvac_system_performance.csv)
```

### Key Modules

| File | Role |
|------|------|
| `main.py` | Orchestration, camera loop, threading, CSV logging |
| `vlm_processor.py` | VLM inference with auto device selection (lcpp→trt→mps→cuda→cpu) |
| `thermal_engine.py` | PMV/PPD calculation (ISO 7730:2005) |
| `control_logic.py` | PMV → HVAC decision (dynamic target temp + PID fan speed) |
| `state_machine.py` | Occupancy state transitions with departure/lunch detection |
| `hvac_simulator.py` | Physical indoor environment simulation |
| `energy_monitor.py` | Power consumption tracking vs. rule-based baseline (1200W fixed) |
| `yolo_detector.py` | YOLOv8n people detection |
| `startup_screen.py` | Environment profile selector (shown on launch) |
| `dashboard.py` | Operator OpenCV window rendering |
| `user_display.py` | User-facing OpenCV window rendering |
| `sensor_interface.py` | SHT31 I2C temperature/humidity sensor (GPIO pins 3/5) |
| `env_profiles.py` | Environment presets (office, home, gym, etc.) |
| `scenario_runner.py` | Offline scenario simulation from JSON files |

### VLM Device Priority
`vlm_processor.py` auto-selects backend at startup:
1. **lcpp** — llama.cpp CUDA INT4 (`~/llama.cpp/build/bin/llama-mtmd-cli` + `libggml-cuda.so`)
2. **trt** — TensorRT engine (`./qwen2vl_engine/`)
3. **mps** — Apple Silicon MPS (HuggingFace)
4. **cuda** — NVIDIA CUDA (HuggingFace)
5. **cpu** — CPU fallback

**Jetson model paths:**
- LLM: `~/llama.cpp/models/Qwen2-VL-2B/qwen2vl-2b-q4km.gguf` (Q4_K_M, 941MB)
- Vision encoder: `~/llama.cpp/models/Qwen2-VL-2B/mmproj-qwen2vl-2b-f16.gguf` (FP16, 1.3GB)

### Energy Estimation Model
`energy_monitor.py` compares AI control vs. rule-based baseline:
- Fan 1/2/3: 800/1200/1600 W
- Baseline: 1200 W constant when occupied
- Comfort rate: fraction of frames with PMV ∈ (-0.5, 0.5)

### CSV Log Schema (`hvac_system_performance.csv`)
Key columns: `timestamp, people_count, activity, clo, met, pmv, ppd, indoor_temp, indoor_humid, outdoor_temp, hvac_on, fan_speed, target_temp, window_open, energy_wh, baseline_wh, comfort_rate`

## Jetson-Specific Notes

**Critical env vars for Jetson:**
- `GGML_CUDA_NO_VMM=1` — prevents CUDA OOM on unified memory
- `LD_PRELOAD` for old board (hanul/JetPack 5.1.2): libgomp + libGLdispatch
- New board (jetson/JetPack 6.x): no LD_PRELOAD needed

**cv2 on Jetson:** PyPI opencv (Qt backend) causes NULL window handler crash. Use system GTK opencv:
```bash
ln -sfn /usr/lib/python3/dist-packages/cv2.cpython-310-aarch64-linux-gnu.so \
    ~/edge-vlm-hvac-system/vlm-env/lib/python3.10/site-packages/cv2.so
```

**pynput on Mac:** Causes HIToolbox crash on macOS 26+. Disabled via `_PYNPUT_OK = platform.system() != "Darwin"`. On Mac, keyboard input falls back to `cv2.waitKey()`.

**llama.cpp build (Jetson Orin Nano Super, CUDA arch 87):**
```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.6
cmake --build build --config Release -j4
```

## Branches

- `main` — stable, Mac-compatible version
- `feature/llamacpp-int4-quantization` — Jetson deployment with llama.cpp CUDA INT4 (use this on Jetson)
