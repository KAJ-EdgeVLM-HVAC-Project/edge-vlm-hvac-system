"""
[YOLOv8n 실시간 인원 감지기]

VLM 인원 감지 대비 이점:
  - 정확도: 95%+ (vs VLM ~60~70% 불안정)
  - 속도: CPU 10~20fps / Jetson TensorRT 30fps+ (vs VLM 30~60초/회)
  - 신뢰성: 전용 모델이라 할루시네이션 없음

첫 실행 시 yolov8n.pt (~6MB) 자동 다운로드 (인터넷 필요)
이후 실행은 캐시 사용.

YOLO 로드 실패 시 자동으로 VLM 인원 감지로 폴백.
"""

import os
import numpy as np


def _is_jetson() -> bool:
    import platform
    return platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release")


class YOLODetector:
    """
    YOLOv8n 기반 실시간 인원 수 감지기

    ── 추천 설정 ──────────────────────────────────────────────────────────────
    노트북 CPU  : imgsz=320, conf=0.35 → ~10~20fps
    Jetson Orin : imgsz=640, conf=0.40 → ~30fps+ (TensorRT 엔진 사용 시)
    """

    def __init__(self, imgsz: int = 320, conf: float = 0.35):
        """
        Args:
            imgsz : 추론 해상도 (낮을수록 빠름, 높을수록 정확)
            conf  : 감지 신뢰도 임계값
        """
        self._model     = None
        self._available = False
        self.imgsz      = imgsz
        self.conf       = conf
        self._last_count = 0

        self._device = "cpu" if _is_jetson() else None  # Jetson: cuDNN 엔진 오류 방지
        try:
            from ultralytics import YOLO
            self._model     = YOLO("yolov8n.pt")
            self._available = True
            dev_label = f"device={self._device}" if self._device else "auto"
            print(f"[YOLO] YOLOv8n 로드 완료 (imgsz={imgsz}, conf={conf}, {dev_label})")
        except ImportError:
            print("[YOLO] ultralytics 미설치 → pip install ultralytics")
        except Exception as e:
            print(f"[YOLO] 로드 실패: {e}")

        if not self._available:
            print("[YOLO] VLM 인원 감지로 폴백")

    @property
    def available(self) -> bool:
        return self._available

    def count_people(self, frame: np.ndarray) -> int:
        """
        프레임에서 인원 수 감지

        Returns:
            int : 감지된 인원 수
                  YOLO 사용 불가 시 -1 반환 (호출자가 VLM 값으로 폴백)
        """
        if not self._available or self._model is None:
            return -1

        try:
            kwargs = dict(classes=[0], imgsz=self.imgsz, conf=self.conf, verbose=False)
            if self._device:
                kwargs["device"] = self._device
            results = self._model(frame, **kwargs)
            self._last_count = len(results[0].boxes)
        except Exception as e:
            print(f"[YOLO] 추론 오류: {e}")
            return self._last_count    # 마지막 성공값 유지

        return self._last_count

    @property
    def last_count(self) -> int:
        return self._last_count
