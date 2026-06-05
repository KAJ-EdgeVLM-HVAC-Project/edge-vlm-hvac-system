"""
[센서 인터페이스 추상화 레이어]

현재 (PC / 개발 단계):
    MODE = 'simulate' → HVACSimulator 가상값 사용

Jetson 이전 시 이 파일만 수정하면 됨:
    MODE = 'sht31'   → I2C SHT31 온습도 센서 (I2C 버스7, 주소 0x44)
    MODE = 'dht22'   → GPIO DHT22 온습도 센서
    MODE = 'bme280'  → I2C BME280 온습도+기압 센서

나머지 모든 코드(main.py, thermal_engine.py 등)는 수정 불필요.
"""
import platform as _platform

def _is_jetson() -> bool:
    return _platform.machine() == "aarch64" and \
           __import__("os").path.exists("/etc/nv_tegra_release")


class SensorInterface:
    """
    실내 온습도 센서 추상화 클래스

    ── 지원 모드 ─────────────────────────────────────────────────────────────
    'simulate' : HVACSimulator 가상값 (기본, 노트북 개발용)
    'sht31'    : I2C SHT31 온습도 센서 (Jetson: 버스7, 주소 0x44)
    'dht22'    : Adafruit DHT22 GPIO 센서 (Jetson 배선 필요)
    'bme280'   : Adafruit BME280 I2C 센서 (Jetson 배선 필요)

    ── Jetson 하드웨어 배선 ──────────────────────────────────────────────────
    SHT31: VCC→핀1(3.3V), GND→핀6, SDA→핀3, SCL→핀5  (I2C 버스7, 0x44)
    """

    MODE     = "sht31" if _is_jetson() else "simulate"
    GPIO_PIN = 4

    SHT31_BUS  = 7     # Jetson Orin Nano Super 40핀 헤더 I2C 버스
    SHT31_ADDR = 0x44  # ADDR 핀 Low 기본값

    def __init__(self, simulator=None):
        """
        Args:
            simulator : HVACSimulator 인스턴스 (simulate 모드에서 사용)
        """
        self._sim = simulator
        self._last_temp  = 20.0
        self._last_humid = 50.0

    def read_climate(self) -> tuple:
        """
        실내 온도(°C)와 습도(%) 반환

        Returns:
            (temp: float, humid: float)
            읽기 실패 시 마지막 성공값 반환
        """
        if self.MODE == "simulate":
            if self._sim is not None:
                self._last_temp  = self._sim.indoor_temp
                self._last_humid = self._sim.indoor_humid
            return self._last_temp, self._last_humid

        elif self.MODE == "sht31":
            try:
                import smbus2
                bus = smbus2.SMBus(self.SHT31_BUS)
                # 단발 측정 명령: High Repeatability, Clock Stretching Off
                bus.write_i2c_block_data(self.SHT31_ADDR, 0x24, [0x00])
                import time as _t; _t.sleep(0.02)
                data = bus.read_i2c_block_data(self.SHT31_ADDR, 0x00, 6)
                bus.close()
                raw_temp  = (data[0] << 8) | data[1]
                raw_humid = (data[3] << 8) | data[4]
                temp  = -45.0 + 175.0 * raw_temp  / 65535.0
                humid =         100.0 * raw_humid / 65535.0
                self._last_temp  = round(temp,  1)
                self._last_humid = round(min(max(humid, 0.0), 100.0), 1)
            except Exception as e:
                print(f"[Sensor] SHT31 읽기 실패: {e} — 이전 값 유지")
            return self._last_temp, self._last_humid

        # ── Jetson DHT22 (아래 주석 해제 후 사용) ────────────────────────────
        # elif self.MODE == "dht22":
        #     try:
        #         import Adafruit_DHT
        #         humid, temp = Adafruit_DHT.read_retry(
        #             Adafruit_DHT.DHT22, self.GPIO_PIN, retries=3)
        #         if temp is not None and humid is not None:
        #             self._last_temp  = round(float(temp),  1)
        #             self._last_humid = round(float(humid), 1)
        #     except Exception as e:
        #         print(f"[Sensor] DHT22 읽기 실패: {e} — 이전 값 유지")
        #     return self._last_temp, self._last_humid

        # ── Jetson BME280 (아래 주석 해제 후 사용) ───────────────────────────
        # elif self.MODE == "bme280":
        #     try:
        #         import board
        #         import adafruit_bme280
        #         i2c = board.I2C()
        #         bme = adafruit_bme280.Adafruit_BME280_I2C(i2c)
        #         self._last_temp  = round(bme.temperature, 1)
        #         self._last_humid = round(bme.relative_humidity, 1)
        #     except Exception as e:
        #         print(f"[Sensor] BME280 읽기 실패: {e} — 이전 값 유지")
        #     return self._last_temp, self._last_humid

        return self._last_temp, self._last_humid

    @property
    def mode(self) -> str:
        return self.MODE
