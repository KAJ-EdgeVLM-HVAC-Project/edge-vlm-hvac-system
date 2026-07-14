"""
[YOLO 실시간 인원 감지기]

Jetson(Orin): TensorRT 엔진(yolo26s.engine)을 **torch 없이** GPU에서 직접 구동.
  - 이 보드의 torch는 Orin(sm_87)용이 아니라 GPU 실행이 안 되므로, ultralytics를
    거치지 않고 tensorrt + cuda-python으로 엔진을 직접 돌린다.
  - 엔진 출력: (1, 300, 6) = [x1,y1,x2,y2,conf,cls] (768 레터박스 좌표, NMS 완료).
  - 엔진이 없거나 로드 실패 시 ultralytics(CPU)로 자동 폴백.
그 외(Mac 등): ultralytics YOLO 사용.

엔진 생성(최초 1회, 보드에서):
  yolo export → onnx → trtexec --onnx=yolo26s.onnx --saveEngine=yolo26s.engine --fp16
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ENGINE = os.path.join(_HERE, "yolo26s.engine")


def _is_jetson() -> bool:
    import platform
    return platform.machine() == "aarch64" and os.path.exists("/etc/nv_tegra_release")


def _letterbox(img, size=768, color=114):
    """비율 유지 리사이즈 + 패딩. 반환: (canvas, ratio, pad_x, pad_y)."""
    import cv2
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def _nms(boxes, iou_thres=0.6):
    """겹치는 박스 제거 (한 사람에 여러 박스 방지). boxes는 conf 내림차순 가정."""
    keep = []
    for b in boxes:
        x1, y1, x2, y2, _ = b
        a = max(0, x2 - x1) * max(0, y2 - y1)
        dup = False
        for k in keep:
            ix1 = max(x1, k[0]); iy1 = max(y1, k[1])
            ix2 = min(x2, k[2]); iy2 = min(y2, k[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ak = max(0, k[2] - k[0]) * max(0, k[3] - k[1])
            if inter / (a + ak - inter + 1e-9) > iou_thres:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


class _TRTYolo:
    """torch 없이 TensorRT 엔진으로 사람(class 0) 검출."""

    def __init__(self, engine_path: str, imgsz: int = 768, conf: float = 0.25,
                 nms_iou: float = 0.6):
        import tensorrt as trt
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart
        self.trt   = trt
        self.cudart = cudart
        self.imgsz = imgsz
        self.conf  = conf
        self.nms_iou = nms_iou

        lg = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(lg) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("엔진 역직렬화 실패")
        self.ctx = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.i_name = next(n for n in names
                           if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.o_name = next(n for n in names
                           if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.ishape = tuple(self.engine.get_tensor_shape(self.i_name))   # (1,3,768,768)
        self.oshape = tuple(self.engine.get_tensor_shape(self.o_name))   # (1,300,6)

        self.h_in  = np.empty(self.ishape, np.float32)
        self.h_out = np.empty(self.oshape, np.float32)
        _, self.d_in  = cudart.cudaMalloc(self.h_in.nbytes)
        _, self.d_out = cudart.cudaMalloc(self.h_out.nbytes)
        _, self.stream = cudart.cudaStreamCreate()
        self.ctx.set_tensor_address(self.i_name, int(self.d_in))
        self.ctx.set_tensor_address(self.o_name, int(self.d_out))

    def infer_person_boxes(self, frame):
        """프레임에서 사람 박스 리스트 반환: [(x1,y1,x2,y2,conf), ...] (원본 좌표)."""
        import cv2
        cudart = self.cudart
        h0, w0 = frame.shape[:2]

        lb, r, dw, dh = _letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        np.copyto(self.h_in, np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]))

        cudart.cudaMemcpyAsync(self.d_in, self.h_in.ctypes.data, self.h_in.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.ctx.execute_async_v3(self.stream)
        cudart.cudaMemcpyAsync(self.h_out.ctypes.data, self.d_out, self.h_out.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        boxes = []
        for x1, y1, x2, y2, c, cls in self.h_out.reshape(-1, 6):
            if c < self.conf:
                break                      # 신뢰도 내림차순 → 이후는 전부 더 낮음
            if int(round(cls)) != 0:       # class 0 = person
                continue
            X1 = min(max((x1 - dw) / r, 0), w0)
            Y1 = min(max((y1 - dh) / r, 0), h0)
            X2 = min(max((x2 - dw) / r, 0), w0)
            Y2 = min(max((y2 - dh) / r, 0), h0)
            boxes.append((int(X1), int(Y1), int(X2), int(Y2), float(c)))
        return _nms(boxes, self.nms_iou)


class YOLODetector:
    """
    실시간 인원 수 감지기 — Jetson은 TensorRT(GPU), 그 외는 ultralytics.

    ── 추천 설정 ──────────────────────────────────────────────────────────────
    Jetson Orin : 엔진(yolo26s.engine, imgsz=768) GPU ~10~15ms
    노트북 CPU  : ultralytics yolo26s, imgsz=320~640
    """

    def __init__(self, imgsz: int = 320, conf: float = 0.25,
                 model_name: str = "yolo26s.pt",
                 engine_path: str = _DEFAULT_ENGINE):
        self._model      = None      # ultralytics 모델 (폴백/비-Jetson)
        self._trt        = None      # TensorRT 백엔드 (Jetson)
        self._available  = False
        self.imgsz       = imgsz
        self.conf        = conf
        self.model_name  = model_name
        self._last_count = 0
        self._last_boxes = []

        # 1) Jetson + 엔진 있으면 TensorRT GPU 우선
        if _is_jetson() and os.path.exists(engine_path):
            try:
                self._trt = _TRTYolo(engine_path, imgsz=768, conf=conf)
                self._available = True
                print(f"[YOLO] TensorRT 엔진(GPU) 로드 완료: {os.path.basename(engine_path)} "
                      f"(imgsz=768, conf={conf})")
            except Exception as e:
                print(f"[YOLO] TensorRT 로드 실패 → ultralytics 폴백: {str(e)[:120]}")
                self._trt = None

        # 2) 폴백/비-Jetson: ultralytics
        if self._trt is None:
            self._device = "cpu" if _is_jetson() else None
            try:
                from ultralytics import YOLO
                self._model     = YOLO(model_name)
                self._available = True
                dev = f"device={self._device}" if self._device else "auto"
                print(f"[YOLO] {model_name} 로드 완료 (imgsz={imgsz}, conf={conf}, {dev})")
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
        """프레임에서 인원 수 감지. 사용 불가 시 -1(호출자가 VLM 폴백)."""
        if not self._available:
            return -1
        try:
            if self._trt is not None:
                self._last_boxes = self._trt.infer_person_boxes(frame)
            else:
                kwargs = dict(classes=[0], imgsz=self.imgsz, conf=self.conf, verbose=False)
                if getattr(self, "_device", None):
                    kwargs["device"] = self._device
                boxes = self._model(frame, **kwargs)[0].boxes
                self._last_boxes = [
                    (int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(c))
                    for b, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist())
                ]
            self._last_count = len(self._last_boxes)
        except Exception as e:
            print(f"[YOLO] 추론 오류: {e}")
            return self._last_count
        return self._last_count

    @property
    def last_count(self) -> int:
        return self._last_count

    @property
    def last_boxes(self) -> list:
        """마지막 감지 박스 [(x1,y1,x2,y2,conf), ...] — 시각화용"""
        return self._last_boxes
