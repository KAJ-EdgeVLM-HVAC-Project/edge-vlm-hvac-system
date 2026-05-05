"""
[NVIDIA Jetson TensorRT 변환 유틸리티]

노트북 개발 완료 후 Jetson Orin Nano Super에서 실행하는 스크립트.
모델을 TensorRT 엔진으로 변환하여 추론 속도를 5~10배 향상시킵니다.

사용법:
    python convert_tensorrt.py --yolo          # YOLOv8n → TRT 엔진 변환
    python convert_tensorrt.py --vlm-guide     # Qwen2-VL 변환 가이드 출력
    python convert_tensorrt.py --all           # 전체 실행

사전 조건 (Jetson):
    - JetPack 6.0+
    - CUDA, TensorRT 설치 확인: python -c "import tensorrt; print(tensorrt.__version__)"
    - ultralytics: pip install ultralytics
"""

import argparse
import os
import sys


def convert_yolo():
    """YOLOv8n → TensorRT FP16 엔진 변환"""
    print("\n[YOLO] YOLOv8n TensorRT 변환 시작...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        # FP16으로 변환 (Jetson GPU 최적화)
        model.export(
            format="engine",
            imgsz=640,
            half=True,       # FP16
            device=0,        # GPU
            simplify=True,
        )
        print("[YOLO] 변환 완료: yolov8n.engine")
        print("       main.py의 YOLODetector에서 'yolov8n.engine' 로드 가능")
    except ImportError:
        print("[YOLO] ultralytics 미설치: pip install ultralytics")
    except Exception as e:
        print(f"[YOLO] 변환 실패: {e}")


def convert_vlm_int4():
    """
    Qwen2-VL-2B → TensorRT-LLM INT4 양자화 변환 (Jetson Orin 전용).

    Step 1: FP16 체크포인트를 INT4 weight-only 양자화로 변환
    Step 2: TRT-LLM 엔진 빌드 (gemm_plugin float16)

    메모리 효과:
      FP32 원본  ~8 GB  → Jetson 8GB UMA 초과 (실행 불가)
      INT4 변환  ~1 GB  → YOLO(0.5GB) + OS(1GB) 포함해도 여유
    """
    commands = [
        (
            "python -m tensorrt_llm.commands.convert_checkpoint "
            "--model_dir Qwen/Qwen2-VL-2B-Instruct "
            "--output_dir ./qwen2vl_ckpt "
            "--dtype float16 "
            "--int4_weight_only_quant"
        ),
        (
            "trtllm-build "
            "--checkpoint_dir ./qwen2vl_ckpt "
            "--output_dir ./qwen2vl_engine "
            "--gemm_plugin float16 "
            "--max_batch_size 1 "
            "--max_input_len 512 "
            "--max_output_len 80"
        ),
    ]

    step_labels = [
        "[Step 1/2] FP16 → INT4 체크포인트 변환",
        "[Step 2/2] TRT-LLM 엔진 빌드",
    ]

    for label, cmd in zip(step_labels, commands):
        print(f"\n{label}")
        print(f"  $ {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            print(f"\n❌ 명령 실패 (exit code {ret}). 위 로그를 확인하세요.")
            sys.exit(ret)

    print("\n✅ INT4 변환 완료: ./qwen2vl_engine")
    print("   vlm_processor.py가 자동으로 TRT 모드로 전환됩니다.")


def vlm_guide():
    """Qwen2-VL TensorRT-LLM 변환 가이드 출력"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Qwen2-VL-2B TensorRT-LLM 변환 가이드 (Jetson Orin)         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. TensorRT-LLM 설치                                        ║
║     pip install tensorrt-llm                                 ║
║                                                              ║
║  2. 모델 가중치 변환 (INT4 양자화)                            ║
║     python -m tensorrt_llm.commands.convert_checkpoint \\    ║
║       --model_dir Qwen/Qwen2-VL-2B-Instruct \\              ║
║       --output_dir ./qwen2vl_ckpt \\                         ║
║       --dtype float16 \\                                      ║
║       --int4_weight_only_quant                               ║
║                                                              ║
║  3. TRT 엔진 빌드                                             ║
║     trtllm-build \\                                           ║
║       --checkpoint_dir ./qwen2vl_ckpt \\                     ║
║       --output_dir ./qwen2vl_engine \\                       ║
║       --gemm_plugin float16 \\                                ║
║       --max_batch_size 1 \\                                   ║
║       --max_input_len 512 \\                                  ║
║       --max_output_len 80                                    ║
║                                                              ║
║  4. vlm_processor.py 수정                                    ║
║     _select_device() → "tensorrt" 모드 추가                  ║
║     모델 로드 → TRT 런타임으로 교체                           ║
║                                                              ║
║  예상 성능 (Jetson Orin Nano Super 8GB):                     ║
║    현재 float32 CPU : 30~60s / 추론                          ║
║    TRT INT4         :  3~8s  / 추론 (약 10배 향상)           ║
╚══════════════════════════════════════════════════════════════╝
""")


def check_jetson():
    """Jetson 환경 확인"""
    print("\n[환경 체크]")
    try:
        with open("/etc/nv_tegra_release") as f:
            print(f"  Jetson: {f.readline().strip()}")
    except FileNotFoundError:
        print("  Jetson 환경 아님 (이 스크립트는 Jetson에서 실행하세요)")
        return False

    import subprocess
    for lib in ["tensorrt", "torch", "ultralytics"]:
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {lib}; print({lib}.__version__)"],
                capture_output=True, text=True, timeout=10,
            )
            ver = result.stdout.strip() or "설치됨"
            print(f"  {lib}: {ver}")
        except Exception:
            print(f"  {lib}: 미설치")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jetson TensorRT 모델 변환 유틸리티"
    )
    parser.add_argument("--yolo",      action="store_true", help="YOLOv8n TRT 변환 (FP16)")
    parser.add_argument("--vlm-int4",  action="store_true", help="Qwen2-VL INT4 양자화 변환 (Jetson 전용)")
    parser.add_argument("--vlm-guide", action="store_true", help="Qwen2-VL 변환 가이드 출력")
    parser.add_argument("--check",     action="store_true", help="Jetson 환경 확인")
    parser.add_argument("--all",       action="store_true", help="전체 실행 (check + yolo + vlm-int4)")
    args = parser.parse_args()

    if args.check or args.all:
        check_jetson()
    if args.yolo or args.all:
        convert_yolo()
    if args.vlm_int4 or args.all:
        convert_vlm_int4()
    if args.vlm_guide:
        vlm_guide()
    if not any(vars(args).values()):
        parser.print_help()
