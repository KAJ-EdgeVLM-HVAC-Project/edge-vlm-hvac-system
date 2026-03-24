import torch
import json
import re
import cv2
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VLMProcessor:
    def __init__(self):
        # 1. 디바이스 설정 (현재 라이젠 노트북은 'cpu', 맥북은 'mps'로 변경 예정)
        self.device = "mps"  # <-- macOS Metal Performance Shaders 최적화됨!
        # self.device = "cpu"    # 현재 윈도우 라이젠 노트북용
        
        print(f"🚀 [VLM] 현재 {self.device.upper()} 모드로 초기화 중...")
        
        self.model_id = "Qwen/Qwen2-VL-2B-Instruct"
        
        try:
            # macOS M5 MPS 최적화: torch_dtype으로 메모리 효율과 성능 극대화
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            print(f"✅ [VLM] {self.device.upper()} 로드 완료")
        except Exception as e:
            print(f"❌ [VLM] 로드 에러: {e}")
            raise

    def analyze_frame(self, frame):
        # [macOS M5 최적화] 성능과 정확도의 균형: 해상도 감소로 분석 속도 개선
        # 384x384는 여전히 충분한 정확도를 유지하면서 처리 속도 ↑
        input_size = 384 
        resized_frame = cv2.resize(frame, (input_size, input_size))
        pil_img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

        # 프롬프트: 더 간단하고 명확한 지시문으로 응답 안정성 향상
        prompt_text = (
            "Analyze this image and respond with ONLY a JSON object: "
            "{\"sleeves\": \"short\" or \"long\", \"outerwear\": \"yes\" or \"no\", "
            "\"activity\": \"sitting\", \"walking\", or \"cooking\", \"people\": number}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

        # 전처리 및 추론
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # M5 칩 성능으로 더 긴 응답 처리 가능 (40 tokens)
            generated_ids = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
            
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        raw_response = output_text[0].split('assistant')[-1].strip()

        try:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # PMV 변수 매핑 (ISO 7730 근거)
                clo = 0.5 if data.get('sleeves') == 'short' else 1.0
                if data.get('outerwear') == 'yes': clo += 0.3
                
                activity_map = {'sitting': 1.0, 'walking': 1.5, 'cooking': 2.0}
                met = activity_map.get(data.get('activity'), 1.2)
                
                return {"clo": clo, "met": met, "count": data.get('people', 1)}
            else:
                print(f"⚠️ [VLM] JSON 파싱 실패 - 응답: {raw_response[:100]}")
        except json.JSONDecodeError as e:
            print(f"❌ [VLM] JSON 디코드 에러: {e}")
            print(f"   원본 응답: {raw_response[:150]}")
        except Exception as e:
            print(f"❌ [VLM] 분석 중 에러: {e}")
            
        return None