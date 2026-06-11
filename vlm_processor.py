import atexit
import base64
import glob
import os
import time
import torch
import json
import re
import platform
import subprocess
import tempfile
import cv2
import requests
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VLMProcessor:
    """
    [Vision Language Model 처리기]
    모델: Qwen3-VL-2B / Qwen2-VL-2B GGUF (Jetson) 또는 Qwen2-VL-2B HF (Mac/PC)
    역할: 카메라 프레임에서 PMV 입력 파라미터 및 맥락 신호를 추출합니다.

    ── 디바이스 우선순위 ───────────────────────────────────────────────────────
      1. LCPP (llama.cpp CUDA INT4 — Jetson, libggml-cuda.so 존재 시) — 최우선
         · llama-server 상주 프로세스 (모델 1회 로드, HTTP 추론) 우선
         · llama-server 불가 시 llama-mtmd-cli subprocess 폴백
      2. MPS  (Apple Silicon M1~M5)              — float16
      3. CUDA (NVIDIA GPU, HuggingFace)          — float16
      4. CPU  (그 외 모든 환경)                   — float16

    ── GGUF 모델 자동 탐색 ─────────────────────────────────────────────────────
    ~/llama.cpp/models/*/ 에서 (모델.gguf + mmproj*.gguf) 쌍을 찾되,
    디렉토리/파일명에 'qwen3'가 포함된 모델을 우선 사용.
    환경변수 LCPP_GGUF / LCPP_MMPROJ 로 직접 지정 가능.

    ── 감지 항목 ──────────────────────────────────────────────────────────────
    PMV 입력:
      sleeves     : 소매 길이 → clo 계산
      outerwear   : 아우터 착용 → clo 보정
      activity    : 활동 분류 → met 변환

    맥락 신호:
      room_size   : 공간 크기 ('small'|'medium'|'large') → 15/30/60 m²
      heat_source : 조리기구 등 열원 → 복사온도(tr) 보정

    ※ 인원 수(people)는 YOLODetector가 전담 — VLM 프롬프트에서 제거됨.
    """

    # llama.cpp CUDA INT4 경로 (Jetson: ~/llama.cpp/build)
    LCPP_BIN    = os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli")
    LCPP_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
    LCPP_LIB    = os.path.expanduser("~/llama.cpp/build/ggml/src/ggml-cuda/libggml-cuda.so")
    LCPP_LIB2   = os.path.expanduser("~/llama.cpp/build/bin/libggml-cuda.so")
    LCPP_MODELS = os.path.expanduser("~/llama.cpp/models")
    LCPP_NGL    = 99      # GPU에 오프로드할 레이어 수 (99 = 전체)
    LCPP_PORT   = 8090    # llama-server 포트 (127.0.0.1 전용)
    LCPP_SERVER_BOOT_SEC = 180  # 서버 기동(모델 로드) 대기 한도

    # llama-server json_schema 강제용 출력 스키마
    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "sleeves":     {"type": "string", "enum": ["long", "short"]},
            "outerwear":   {"type": "string", "enum": ["yes", "no"]},
            "activity":    {"type": "string", "enum": ["lying", "sitting", "standing",
                                                       "walking", "cooking", "exercising"]},
            "room_size":   {"type": "string", "enum": ["small", "medium", "large"]},
            "heat_source": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["sleeves", "outerwear", "activity", "room_size", "heat_source"],
    }

    # PMV 입력 매핑 테이블 (ISO 7730:2005 근거)
    CLO_BASE  = {'short': 0.5, 'long': 1.0}
    CLO_OUTER = 0.3   # 아우터 착용 시 추가 착의량

    ROOM_SIZE_MAP = {'small': 15.0, 'medium': 30.0, 'large': 60.0}  # m²

    MET_MAP = {
        'lying':      0.8,   # 누워있음 (수면/휴식)
        'sitting':    1.0,   # 착석 (사무 작업)
        'standing':   1.2,   # 기립 (가벼운 활동)
        'walking':    1.5,   # 보행
        'cooking':    2.0,   # 조리 (서서 작업)
        'exercising': 3.0,   # 운동 (유산소)
    }
    MET_DEFAULT = 1.2   # 분류 불가 시 기립 수준
    TR_HEAT_OFFSET = 4.0  # 열원 감지 시 복사온도 보정값 (°C)

    @staticmethod
    def _find_lcpp_model():
        """GGUF (모델, mmproj) 쌍 자동 탐색.

        우선순위:
          1. 환경변수 LCPP_GGUF + LCPP_MMPROJ
          2. ~/llama.cpp/models/*/ 중 'qwen3' 포함 디렉토리
          3. 그 외 (모델+mmproj 쌍이 있는 첫 디렉토리)

        Returns:
            (gguf_path, mmproj_path) 또는 (None, None)
        """
        env_gguf   = os.environ.get("LCPP_GGUF")
        env_mmproj = os.environ.get("LCPP_MMPROJ")
        if env_gguf and env_mmproj and \
           os.path.exists(env_gguf) and os.path.exists(env_mmproj):
            return env_gguf, env_mmproj

        candidates = []   # (priority, gguf, mmproj)
        for d in sorted(glob.glob(os.path.join(VLMProcessor.LCPP_MODELS, "*"))):
            if not os.path.isdir(d):
                continue
            ggufs   = sorted(glob.glob(os.path.join(d, "*.gguf")))
            mmprojs = [g for g in ggufs if "mmproj" in os.path.basename(g).lower()]
            models  = [g for g in ggufs if "mmproj" not in os.path.basename(g).lower()]
            if not mmprojs or not models:
                continue
            # Q4 양자화 모델 우선
            q4 = [m for m in models if "q4" in os.path.basename(m).lower()]
            model = q4[0] if q4 else models[0]
            prio  = 0 if "qwen3" in os.path.basename(d).lower() else 1
            candidates.append((prio, model, mmprojs[0]))

        if not candidates:
            return None, None
        candidates.sort(key=lambda c: c[0])
        return candidates[0][1], candidates[0][2]

    @staticmethod
    def _lcpp_cuda_available() -> bool:
        """llama.cpp CUDA 백엔드 실행 가능 여부 확인."""
        cuda_lib = (os.path.exists(VLMProcessor.LCPP_LIB) or
                    os.path.exists(VLMProcessor.LCPP_LIB2))
        if not cuda_lib:
            return False
        has_bin = (os.path.exists(VLMProcessor.LCPP_BIN) or
                   os.path.exists(VLMProcessor.LCPP_SERVER))
        gguf, mmproj = VLMProcessor._find_lcpp_model()
        return has_bin and gguf is not None

    @staticmethod
    def _select_device():
        """
        최적 추론 디바이스 자동 선택
          - llama.cpp CUDA INT4 (libggml-cuda.so 존재): 'lcpp', float16  (최우선)
          - Apple Silicon (MPS 사용 가능):               'mps',  float16
          - NVIDIA GPU (CUDA 사용 가능):                 'cuda', float16
          - CPU 전용 또는 그 외:                         'cpu',  float16
        """
        if VLMProcessor._lcpp_cuda_available():
            return "lcpp", torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float16

    def __init__(self):
        self.device, self.dtype = self._select_device()
        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"

        chip = platform.processor() or platform.machine()
        print(f"🚀 [VLM] {self.device.upper()} ({chip}) 모드로 초기화 중...")

        try:
            if self.device == "lcpp":
                # llama.cpp CUDA INT4 — Jetson에서 GPU 직접 추론
                # libggml-cuda.so 위치를 LD_LIBRARY_PATH에 추가하여 동적 로드
                lib_dir = os.path.dirname(self.LCPP_LIB if os.path.exists(self.LCPP_LIB)
                                          else self.LCPP_LIB2)
                bin_dir = os.path.dirname(self.LCPP_BIN)
                self._lcpp_lib_dir = f"{lib_dir}:{bin_dir}"
                self._lcpp_gguf, self._lcpp_mmproj = self._find_lcpp_model()
                self.model     = None  # llama.cpp 방식 — HF 모델 객체 없음
                self.processor = None
                self._server_proc = None

                print(f"✅ [VLM] llama.cpp CUDA INT4 준비 완료")
                print(f"   모델: {self._lcpp_gguf}")
                print(f"   mmproj: {self._lcpp_mmproj}")
                print(f"   GPU 레이어: {self.LCPP_NGL}")

                # llama-server 상주 프로세스 기동 시도 (모델 1회 로드 → 추론 수 초)
                # 실패하면 기존 llama-mtmd-cli subprocess 방식으로 폴백
                if os.path.exists(self.LCPP_SERVER):
                    self._start_lcpp_server()
                else:
                    print("ℹ️ [VLM] llama-server 바이너리 없음 — CLI 모드 사용 "
                          "(매 추론마다 모델 재로딩, 느림)")
                return  # 아래 HF 로드 건너뜀

            elif self.device == "mps":
                # Apple Silicon: device_map 미사용, 로드 후 .to('mps')
                # attn_implementation="eager": MPS SDPA 차원 버그 우회
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    local_files_only=True,
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, local_files_only=True
                )
                print(f"✅ [VLM] {self.model_id} 로드 완료 "
                      f"(device={self.device}, dtype={self.dtype})")
            else:
                # CUDA / CPU: device_map으로 직접 배치
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    device_map={"": self.device},
                    local_files_only=True,
                )
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, local_files_only=True
                )
                print(f"✅ [VLM] {self.model_id} 로드 완료 "
                      f"(device={self.device}, dtype={self.dtype})")
        except Exception as e:
            print(f"❌ [VLM] 모델 로드 실패: {e}")
            self.model     = None
            self.processor = None

    # ── llama-server 상주 프로세스 ────────────────────────────────────────────

    def _lcpp_env(self) -> dict:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            self._lcpp_lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        # Jetson 통합 메모리에서 CUDA VMM이 OOM을 유발 → cudaMalloc 방식으로 강제
        env["GGML_CUDA_NO_VMM"] = "1"
        return env

    def _start_lcpp_server(self):
        """llama-server를 백그라운드로 띄우고 /health 응답까지 대기."""
        cmd = [
            self.LCPP_SERVER,
            "-m",       self._lcpp_gguf,
            "--mmproj", self._lcpp_mmproj,
            "-ngl",     str(self.LCPP_NGL),
            "--host",   "127.0.0.1",
            "--port",   str(self.LCPP_PORT),
            "--no-warmup",   # warmup이 최대 해상도로 OOM 유발
        ]
        log_path = os.path.join(tempfile.gettempdir(), "llama-server-hvac.log")
        try:
            self._server_log = open(log_path, "w")
            self._server_proc = subprocess.Popen(
                cmd, env=self._lcpp_env(),
                stdout=self._server_log, stderr=subprocess.STDOUT,
            )
            atexit.register(self.close)
        except Exception as e:
            print(f"⚠️ [VLM] llama-server 기동 실패: {e} — CLI 모드로 폴백")
            self._server_proc = None
            return

        print(f"⏳ [VLM] llama-server 기동 중 (최대 {self.LCPP_SERVER_BOOT_SEC}초)...")
        deadline = time.time() + self.LCPP_SERVER_BOOT_SEC
        url = f"http://127.0.0.1:{self.LCPP_PORT}/health"
        while time.time() < deadline:
            if self._server_proc.poll() is not None:
                print(f"⚠️ [VLM] llama-server 비정상 종료 (로그: {log_path}) — CLI 모드로 폴백")
                self._server_proc = None
                return
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    print(f"✅ [VLM] llama-server 준비 완료 (port {self.LCPP_PORT}, "
                          f"로그: {log_path})")
                    return
            except requests.RequestException:
                pass
            time.sleep(1.0)

        print(f"⚠️ [VLM] llama-server 기동 타임아웃 — CLI 모드로 폴백")
        self.close()

    def close(self):
        """llama-server 프로세스 종료 (atexit 및 수동 정리용)."""
        proc = getattr(self, "_server_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._server_proc = None
        log = getattr(self, "_server_log", None)
        if log is not None and not log.closed:
            log.close()

    def _analyze_frame_lcpp_server(self, frame):
        """llama-server HTTP API로 프레임 분석 (json_schema 강제)."""
        h, w = frame.shape[:2]
        if w > 640 or h > 480:
            frame = cv2.resize(frame, (640, 480))
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return self._default_result()
        b64 = base64.b64encode(jpg.tobytes()).decode()

        prompt = (
            "Look at the image and classify:\n"
            "sleeves: long or short\n"
            "outerwear: yes or no\n"
            "activity: lying, sitting, standing, walking, cooking, exercising\n"
            "room_size: small, medium, large\n"
            "heat_source: yes or no (stove, heater, oven, open flame)\n"
            "Answer in JSON."
        )
        payload = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens":  120,
            "temperature": 0.3,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "hvac_context", "schema": self.JSON_SCHEMA},
            },
        }
        try:
            r = requests.post(
                f"http://127.0.0.1:{self.LCPP_PORT}/v1/chat/completions",
                json=payload, timeout=120,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            print(f"[VLM OUTPUT]\n{raw}\n", flush=True)
            return self._parse_response(raw)
        except Exception as e:
            print(f"⚠️ [VLM-SERVER] 추론 실패: {e} — CLI 모드로 폴백")
            self.close()
            return self._analyze_frame_lcpp(frame)

    def analyze_frame(self, frame):
        """
        프레임 분석 → PMV 입력 파라미터 + 맥락 신호 반환

        Returns:
            dict: {
                'clo': float,          착의량 (ISO 7730)
                'met': float,          대사율 (ISO 7730)
                'room_size': str,      공간 크기 ('small'|'medium'|'large')
                'room_size_m2': float, 공간 면적 (m²)
                'heat_source': str,    열원 존재 ('yes'|'no')
                'outerwear': str,      아우터 착용 ('yes'|'no')
                'activity': str,       활동 분류 원문
            }
            None: 분석 실패 시
            ※ 인원 수는 YOLODetector.count_people()에서 별도 반환
        """
        if self.device == "lcpp":
            if self._server_proc is not None and self._server_proc.poll() is None:
                return self._analyze_frame_lcpp_server(frame)
            return self._analyze_frame_lcpp(frame)

        if self.model is None or self.processor is None:
            print("⚠️ [VLM] 모델이 로드되지 않아 분석 불가.")
            return None

        # 원본 해상도 그대로 사용 (640×480). Qwen2-VL은 가변 해상도 입력 지원.
        # 320×320 다운스케일은 4:3 비율을 정방형으로 왜곡하고 원거리 피사체 픽셀을 손실시킴.
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        prompt_text = (
            "Fill in the JSON below. Output ONLY the JSON, no other text.\n"
            '{"sleeves":"___","outerwear":"___","activity":"___","room_size":"___","heat_source":"___"}\n'
            "sleeves: long or short\n"
            "outerwear: yes or no\n"
            "activity: lying, sitting, standing, walking, cooking, exercising\n"
            "room_size: small, medium, large\n"
            "heat_source: yes or no"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text": prompt_text},
                ],
            },
        ]

        text            = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text           += '{"sleeves":"'  # prefix forcing: 첫 키까지 고정해 JSON 구조 이탈 방지
        image_inputs, _ = process_vision_info(messages)
        inputs          = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        )

        # MPS/CUDA: pixel_values를 모델 dtype(float16)으로 캐스팅 후 디바이스 이동
        if self.device in ("mps", "cuda") and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.3,
            )

        # 입력 토큰 수 계산 (입력 제외하고 새로 생성된 부분만 디코딩)
        # prefix forcing으로 추가한 "{" 를 앞에 다시 붙여서 완전한 JSON으로 복원
        input_len    = inputs["input_ids"].shape[1]
        new_tokens   = generated_ids[:, input_len:]
        output_text  = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        raw_response = '{"sleeves":"' + output_text[0].strip()

        return self._parse_response(raw_response)

    def _analyze_frame_lcpp(self, frame):
        """llama.cpp CUDA INT4로 프레임 분석 (Jetson GPU 전용)."""
        # 프롬프트를 {"sleeves":" 로 끝내 llama.cpp가 JSON을 이어서 생성하도록 강제
        prompt = (
            "Fill in the JSON. Output ONLY the JSON, no other text.\n"
            "sleeves: long or short\n"
            "outerwear: yes or no\n"
            "activity: lying, sitting, standing, walking, cooking, exercising\n"
            "room_size: small, medium, large\n"
            "heat_source: yes or no\n"
            '{"sleeves":"'
        )

        tmp_img = None
        try:
            # 프레임을 임시 JPEG 파일로 저장 (llama.cpp CLI 입력)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_img = f.name
            # VLM 입력 이미지는 640x480으로 축소 (추론 속도 개선)
            h, w = frame.shape[:2]
            if w > 640 or h > 480:
                frame = cv2.resize(frame, (640, 480))
            cv2.imwrite(tmp_img, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            cmd = [
                self.LCPP_BIN,
                "-m",      self._lcpp_gguf,
                "--mmproj", self._lcpp_mmproj,
                "-ngl",    str(self.LCPP_NGL),
                "--image", tmp_img,
                "-p",      prompt,
                "-n",      "80",
                "--temp",  "0.3",
                "--repeat-penalty", "1.3",
                "--no-warmup",       # warmup이 최대 해상도(1288×1288)로 OOM 유발
                "--log-disable",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                env=self._lcpp_env(),
            )

            # stdout에서 ggml/llama 로그 및 echo된 프롬프트 라인 제거
            def _clean(text: str) -> str:
                import re as _re
                lines = []
                for line in text.splitlines():
                    l = line.strip()
                    if not l:
                        continue
                    if any(l.startswith(p) for p in (
                        'E ggml', 'W ggml', 'I ggml', 'ggml_',
                        'warning:', 'llama_', 'clip_', 'encode_',
                        'main:', 'Log ',
                        # echo된 프롬프트 라인 제거
                        'Fill in', 'sleeves:', 'outerwear:', 'activity:',
                        'room_size:', 'heat_source:', 'Output ONLY',
                    )):
                        continue
                    # lcpp 타이밍/통계 라인 (숫자로 시작) 필터
                    if _re.match(r'^\d+[\.\d]*\s', l):
                        continue
                    # CUDA 에러 키워드 포함 라인 필터
                    if any(kw in l for kw in (
                        'ggml_cuda_init', 'failed to initialize CUDA',
                        'CUDA-capable device', 'no CUDA',
                    )):
                        continue
                    lines.append(l)
                return ' '.join(lines).strip()

            raw = _clean(result.stdout)
            if not raw:
                raw = _clean(result.stderr)

            # 프롬프트 prefix와 함께 완전한 JSON 복원
            raw = '{"sleeves":"' + raw
            print(f"[VLM OUTPUT]\n{raw}\n", flush=True)
            return self._parse_response(raw)

        except subprocess.TimeoutExpired:
            print("⚠️ [VLM-LCPP] 추론 타임아웃 (120s)")
            return self._default_result()
        except Exception as e:
            print(f"⚠️ [VLM-LCPP] 오류: {e}")
            return self._default_result()
        finally:
            if tmp_img and os.path.exists(tmp_img):
                os.unlink(tmp_img)

    def _default_result(self, raw=""):
        """모델 거절/파싱 실패 시 반환할 기본값"""
        return {
            "raw_response": raw,
            "sleeves":      "long",
            "clo":          1.0,
            "met":          self.MET_DEFAULT,
            "room_size":    "medium",
            "room_size_m2": 30.0,
            "heat_source":  "no",
            "outerwear":    "no",
            "activity":     "standing",
        }

    def _parse_response(self, raw_response: str):
        """VLM 응답 파싱 및 PMV 파라미터 + 맥락 신호 매핑.
        JSON 파싱 실패 시 자연어 키워드 매핑으로 fallback.
        """
        try:
            # 텍스트 내 모든 {...} 패턴 시도 — lcpp가 프롬프트를 echo할 때
            # 앞쪽 {가 잘못 매칭되는 문제 방지를 위해 마지막 유효 JSON 우선 사용
            data = None
            for m in re.finditer(r'\{[^{}]+\}', raw_response):
                try:
                    candidate = json.loads(m.group())
                    if any(k in candidate for k in ('sleeves', 'activity', 'outerwear')):
                        data = candidate  # 마지막으로 유효한 것을 덮어씀
                except json.JSONDecodeError:
                    continue
            if data is None:
                # JSON 없으면 regex로 enum값 직접 추출 시도
                data = {}
                sm = re.search(r'"sleeves"\s*:\s*"(long|short)"', raw_response)
                am = re.search(r'"activity"\s*:\s*"(sitting|standing|walking|lying|cooking|exercising)"', raw_response)
                om = re.search(r'"outerwear"\s*:\s*"(yes|no)"', raw_response)
                rm = re.search(r'"room_size"\s*:\s*"(small|medium|large)"', raw_response)
                hm = re.search(r'"heat_source"\s*:\s*"(yes|no)"', raw_response)
                if sm: data['sleeves']     = sm.group(1)
                if am: data['activity']    = am.group(1)
                if om: data['outerwear']   = om.group(1)
                if rm: data['room_size']   = rm.group(1)
                if hm: data['heat_source'] = hm.group(1)
                if not data:
                    # 최후 fallback: 자연어 키워드 매핑
                    data = self._extract_from_text(raw_response)
                    print(f"[VLM] 자연어 파싱. 응답: {raw_response[:80]}")
                else:
                    print(f"[VLM] regex 파싱 성공: {data}")

            clo = self.CLO_BASE.get(data.get('sleeves', 'long'), 1.0)
            if data.get('outerwear') == 'yes':
                clo += self.CLO_OUTER

            activity     = data.get('activity', 'standing')
            met          = self.MET_MAP.get(activity, self.MET_DEFAULT)
            room_size    = data.get('room_size', 'medium')
            room_size_m2 = self.ROOM_SIZE_MAP.get(room_size, 30.0)

            return {
                "raw_response": raw_response,
                "sleeves":      data.get('sleeves', 'long'),
                "clo":          round(clo, 2),
                "met":          met,
                "room_size":    room_size,
                "room_size_m2": room_size_m2,
                "heat_source":  data.get('heat_source', 'no'),
                "outerwear":    data.get('outerwear', 'no'),
                "activity":     activity,
            }

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"⚠️ [VLM] 파싱 실패: {e} | 응답: {raw_response[:80]}")
            return self._default_result(raw=raw_response)

    def _extract_from_text(self, text: str) -> dict:
        """자연어 응답에서 키워드로 JSON 필드 추출"""
        t = text.lower()
        data = {}

        # sleeves
        if any(w in t for w in ['short sleeve', 't-shirt', 'tshirt', 'tank top', 'short-sleeve']):
            data['sleeves'] = 'short'
        else:
            data['sleeves'] = 'long'

        # outerwear
        if any(w in t for w in ['jacket', 'coat', 'hoodie', 'overcoat', 'blazer', 'cardigan']):
            data['outerwear'] = 'yes'
        else:
            data['outerwear'] = 'no'

        # activity
        if any(w in t for w in ['lying', 'lying down', 'sleeping']):
            data['activity'] = 'lying'
        elif any(w in t for w in ['walking', 'moving', 'pacing']):
            data['activity'] = 'walking'
        elif any(w in t for w in ['standing', 'stood', 'stand up']):
            data['activity'] = 'standing'
        elif any(w in t for w in ['cooking', 'kitchen', 'stove', 'frying']):
            data['activity'] = 'cooking'
        elif any(w in t for w in ['exercising', 'workout', 'gym', 'running']):
            data['activity'] = 'exercising'
        else:
            data['activity'] = 'sitting'

        # room_size
        if any(w in t for w in ['large room', 'big room', 'spacious', 'hall', 'gym', 'auditorium']):
            data['room_size'] = 'large'
        elif any(w in t for w in ['small room', 'tiny', 'closet', 'narrow']):
            data['room_size'] = 'small'
        else:
            data['room_size'] = 'medium'

        # heat_source
        if any(w in t for w in ['stove', 'heater', 'oven', 'fire', 'furnace', 'heat source']):
            data['heat_source'] = 'yes'
        else:
            data['heat_source'] = 'no'

        return data
