import cv2
import time
import os
import pandas as pd
from datetime import datetime

# 우리가 만든 모듈들 임포트
from vlm_processor import VLMProcessor
from weather_service import WeatherService
from hvac_simulator import HVACSimulator
from thermal_engine import ThermalEngine

# --- [추가] 로그 관련 설정 ---
LOG_FILE = "hvac_system_performance.csv"
SCENARIO_NAME = "Smart_Office_Initial_Test" # 현재 시나리오 명칭

def initialize_csv():
    """CSV 파일이 없으면 헤더를 포함하여 생성"""
    if not os.path.exists(LOG_FILE):
        columns = [
            'timestamp', 'scenario', 'out_temp', 'in_temp', 'in_humid', 
            'people_count', 'met', 'clo', 'pmv_val', 'comfort_status', 'target_temp', 'fan_speed'
        ]
        pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)
        print(f"📁 [System] 새 로그 파일 생성 완료: {LOG_FILE}")

def save_log(data):
    """딕셔너리 데이터를 CSV에 추가"""
    df = pd.DataFrame([data])
    df.to_csv(LOG_FILE, mode='a', index=False, header=False)

def main():
    # --- [수정] 날씨 API 설정 ---
    WEATHER_API_KEY = "97e9ad342e69a006e6c55886b18842c2"
    WEATHER_CITY = "Busan"  # Saha-gu는 동네명이므로 도시명으로 변경

    # 1. 각 모듈 및 로그 초기화
    print("⚙️ [System] 지능형 공조 제어 시스템을 초기화합니다...")
    initialize_csv() # 로그 파일 준비
    
    vlm = VLMProcessor()
    weather = WeatherService(city=WEATHER_CITY, api_key=WEATHER_API_KEY) 
    hvac = HVACSimulator()
    engine = ThermalEngine()
    
    # 2. 카메라 설정 (노트북 내장 카메라만 사용)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # 640 -> 480: 해상도 조정으로 프레임 처리 속도 개선
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # 480 -> 360: 버벅거림 해소
    if not cap.isOpened():
        print("❌ [Error] 카메라를 열 수 없습니다.")
        return

    print("\n✅ 모든 모듈 준비 완료!")
    print(f"📍 현재 날씨 조회 지역: {WEATHER_CITY}")
    print("⌨️  's': 분석 및 제어 실행 | 'q': 프로그램 종료")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 화면 표시용 텍스트
        status_text = f"Temp: {hvac.indoor_temp:.1f}C | Humid: {hvac.indoor_humid:.1f}% | AC: {'ON' if hvac.is_on else 'OFF'}"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('VLM Intelligent HVAC System', frame)

        # 실시간 날씨 반영 및 시뮬레이션 업데이트
        out_temp, _ = weather.fetch_current_weather()
        hvac.simulate_step(out_temp)

        key = cv2.waitKey(30) & 0xFF  # 30ms 대기하여 키입력 감지 확률 향상

        # --- 핵심 제어 및 로그 기록 루프 ('s' 키) ---
        if key == ord('s') or key == ord('S'):
            print("\n🔍 [Step 1] VLM 시각 분석 시작...")
            vlm_data = vlm.analyze_frame(frame)
            
            if vlm_data:
                print(f"📊 [Step 2] 분석 결과: Met({vlm_data['met']}), Clo({vlm_data['clo']}), 인원({vlm_data['count']})")
                
                # PMV 계산
                pmv_val = engine.calculate_pmv(
                    ta=hvac.indoor_temp, 
                    tr=hvac.indoor_temp, 
                    rh=hvac.indoor_humid, 
                    vel=0.1, 
                    met=vlm_data['met'], 
                    clo=vlm_data['clo']
                )
                
                comfort_msg = engine.get_comfort_status(pmv_val)
                print(f"🌡️ [Step 3] 현재 PMV 지수: {pmv_val:.2f} ({comfort_msg})")

                # 제어 로직 및 로그용 변수 설정
                target_temp = 24.0
                fan_speed = 1
                
                print("🎮 [Step 4] 공조기 제어 명령 하달")
                if pmv_val > 0.5:
                    print("🥵 더운 상태 -> 냉방 강화")
                    target_temp, fan_speed = 22.0, 3
                    hvac.set_control(power=True, target=target_temp, fan=fan_speed)
                elif pmv_val < -0.5:
                    print("🥶 추운 상태 -> 난방/절전 모드")
                    target_temp, fan_speed = 26.0, 1
                    hvac.set_control(power=True, target=target_temp, fan=fan_speed)
                else:
                    print("✅ 쾌적 상태 -> 현재 유지")

                # --- [추가] 로그 데이터 저장 ---
                log_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'scenario': SCENARIO_NAME,
                    'out_temp': out_temp,
                    'in_temp': hvac.indoor_temp,
                    'in_humid': hvac.indoor_humid,
                    'people_count': vlm_data['count'],
                    'met': vlm_data['met'],
                    'clo': vlm_data['clo'],
                    'pmv_val': round(pmv_val, 3),
                    'comfort_status': comfort_msg,
                    'target_temp': target_temp,
                    'fan_speed': fan_speed
                }
                save_log(log_data)
                print(f"💾 [Log] 데이터가 {LOG_FILE}에 저장되었습니다.")

            else:
                print("⚠️ [Warning] VLM 분석에 실패했습니다.")

        elif key == ord('q') or key == ord('Q'):
            print("\n👋 프로그램을 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()