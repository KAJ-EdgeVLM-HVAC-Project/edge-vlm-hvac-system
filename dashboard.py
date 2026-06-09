"""
[대시보드 UI 모듈]
카메라 화면 옆에 표시될 정보 패널을 PIL로 렌더링합니다.
한글/영문 혼용 폰트 자동 감지 (Windows: 맑은고딕, macOS: AppleSDGothicNeo, Linux: NanumGothic)
"""

import os
import platform
import cv2
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from state_machine import SystemState

# ── 패널 크기 ─────────────────────────────────────────────────────────────────
PANEL_W = 680

# ── 색상 팔레트 (RGB) ─────────────────────────────────────────────────────────
BG        = ( 22,  22,  32)   # 전체 배경
BG_SECT   = ( 32,  32,  46)   # 섹션 배경
BG_HDR    = ( 52,  48,  75)   # 섹션 헤더 배경
C_TITLE   = (185, 165, 255)   # 섹션 타이틀
C_LABEL   = (125, 125, 150)   # 라벨 (좌측)
C_VAL     = (215, 215, 228)   # 일반 값
C_GREEN   = ( 90, 210,  90)   # 쾌적/정상
C_ORANGE  = (255, 172,  55)   # 경고
C_RED     = (215,  72,  72)   # 위험
C_HEAT    = (255, 148,  55)   # 난방 모드
C_COOL    = ( 72, 165, 255)   # 냉방 모드
C_GOLD    = (255, 222,  88)   # 솔루션 텍스트
C_CYAN    = ( 95, 215, 208)   # 날씨 정보
C_TIME    = (140, 138, 170)   # 타임스탬프

# ── 폰트 캐시 ─────────────────────────────────────────────────────────────────
_font_cache: dict = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """OS별 한글 지원 폰트 자동 로드 (캐싱)"""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]

    sys_name = platform.system()
    if sys_name == 'Windows':
        root = os.environ.get('SystemRoot', r'C:\Windows')
        cands = [
            os.path.join(root, 'Fonts', 'malgunbd.ttf' if bold else 'malgun.ttf'),
            os.path.join(root, 'Fonts', 'malgun.ttf'),
            os.path.join(root, 'Fonts', 'gulim.ttc'),
        ]
    elif sys_name == 'Darwin':
        cands = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/Library/Fonts/NanumGothic.ttf',
        ]
    else:  # Linux / Jetson
        cands = [
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf' if bold
            else '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]

    for path in cands:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[key] = f
            return f
        except Exception:
            pass

    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ── 색상 헬퍼 ─────────────────────────────────────────────────────────────────

def _pmv_color(pmv: float) -> tuple:
    if -0.5 <= pmv <= 0.5:
        return C_GREEN
    if -1.5 <= pmv <= 1.5:
        return C_ORANGE
    return C_RED


def _temp_color(temp: float, is_outdoor: bool = True) -> tuple:
    if is_outdoor:
        if temp < 5:   return C_COOL
        if temp < 15:  return C_VAL
        if temp < 28:  return C_GREEN
        return C_RED
    else:
        if temp < 18:  return C_COOL
        if temp < 27:  return C_GREEN
        return C_RED


# ── 섹션 드로잉 헬퍼 ──────────────────────────────────────────────────────────

def _sect_header(draw: ImageDraw.Draw, y: int, title: str) -> int:
    draw.rectangle([(0, y), (PANEL_W, y + 40)], fill=BG_HDR)
    draw.text((14, y + 8), title, font=_font(19, bold=True), fill=C_TITLE)
    return y + 40


def _row2(draw: ImageDraw.Draw, y: int,
          lbl1: str, val1: str, col1: tuple,
          lbl2: str, val2: str, col2: tuple) -> int:
    draw.text(( 14, y + 4), lbl1, font=_font(17), fill=C_LABEL)
    draw.text((100, y + 4), val1, font=_font(19, bold=True), fill=col1)
    draw.text((350, y + 4), lbl2, font=_font(17), fill=C_LABEL)
    draw.text((440, y + 4), val2, font=_font(19, bold=True), fill=col2)
    return y + 34


def _row1(draw: ImageDraw.Draw, y: int,
          lbl: str, val: str, col: tuple = None) -> int:
    draw.text((14, y + 4), lbl, font=_font(17), fill=C_LABEL)
    draw.text((100, y + 4), val, font=_font(19), fill=col or C_VAL)
    return y + 34


# ── 솔루션 텍스트 생성 ────────────────────────────────────────────────────────

def _get_solution(state: SystemState, pmv: float, hvac,
                  out_temp: float, people: int) -> list:
    """현재 상황에 맞는 솔루션 2줄 반환"""
    mode_str = "난방" if hvac.mode == 'heat' else "냉방"
    on_str   = "ON" if hvac.is_on else "OFF"

    if state == SystemState.EMPTY:
        return ["공실 감지 — 에어컨 OFF", "에너지 절약 대기 모드"]

    if state == SystemState.PRE_DEPARTURE:
        return ["퇴근 준비 맥락 감지!", "절전 모드 전환 — Fan 1 유지"]

    # PMV 기반 메시지 (ARRIVAL / STEADY 공통)
    if pmv > 1.5:
        pmv_msg = f"PMV {pmv:+.2f} — 매우 더움!"
        act_msg = f"{mode_str} {on_str} · Fan {hvac.fan_speed} · 목표 {hvac.target_temp:.0f}°C"
    elif pmv > 0.5:
        pmv_msg = f"PMV {pmv:+.2f} — 조금 더움"
        act_msg = f"냉방 강화 중 → 목표 {hvac.target_temp:.0f}°C"
    elif pmv < -1.5:
        pmv_msg = f"PMV {pmv:+.2f} — 매우 추움!"
        act_msg = f"{mode_str} {on_str} · Fan {hvac.fan_speed} · 목표 {hvac.target_temp:.0f}°C"
    elif pmv < -0.5:
        pmv_msg = f"PMV {pmv:+.2f} — 조금 추움"
        act_msg = f"난방 강화 중 → 목표 {hvac.target_temp:.0f}°C"
    else:
        pmv_msg = f"PMV {pmv:+.2f} — 쾌적 상태"
        act_msg = f"최적 열환경 유지 중 · {hvac.indoor_temp:.1f}°C"

    if state == SystemState.ARRIVAL:
        return [f"[도착] {pmv_msg}", act_msg]

    return [pmv_msg, act_msg]


# ── 섹션별 드로잉 함수 ────────────────────────────────────────────────────────

def _draw_header(draw: ImageDraw.Draw, y: int) -> int:
    draw.rectangle([(0, y), (PANEL_W, y + 64)], fill=(40, 38, 62))
    draw.text((14,  y + 6),  'VLM HVAC SYSTEM',
              font=_font(24, bold=True), fill=C_TITLE)
    draw.text((14, y + 40), datetime.now().strftime('%Y-%m-%d  %H:%M:%S'),
              font=_font(16), fill=C_TIME)
    return y + 64


def _pm_color(pm10: int) -> tuple:
    if pm10 <= 30:   return C_GREEN
    if pm10 <= 80:   return C_ORANGE
    return C_RED


def _draw_outdoor(draw: ImageDraw.Draw, y: int,
                  temp: float, humid: float,
                  weather: str, wind: float,
                  ds: dict = None) -> int:
    draw.rectangle([(0, y), (PANEL_W, y + 180)], fill=BG_SECT)
    y = _sect_header(draw, y, '  실외 환경')
    y = _row2(draw, y,
              '기온',  f'{temp:.1f}°C',  _temp_color(temp, True),
              '습도',  f'{humid:.0f}%',  C_VAL)
    y = _row1(draw, y, '날씨', weather[:24], C_CYAN)
    y = _row1(draw, y, '풍속', f'{wind:.1f} m/s')
    pm10 = int(ds.get('pm10', 0)) if ds else 0
    pm25 = int(ds.get('pm25', 0)) if ds else 0
    y = _row2(draw, y,
              'PM10', f'{pm10} ㎍/㎥', _pm_color(pm10),
              'PM2.5', f'{pm25} ㎍/㎥', _pm_color(pm25))
    return y


def _khai_str(khai) -> tuple:
    """통합대기환경지수 문자열 + 색상"""
    try:
        v = int(khai)
    except (TypeError, ValueError):
        return '-', C_LABEL
    if v <= 1:   return '좋음', C_GREEN
    if v <= 2:   return '보통', C_VAL
    if v <= 3:   return '나쁨', C_ORANGE
    return '매우나쁨', C_RED


def _draw_indoor(draw: ImageDraw.Draw, y: int, hvac, ds: dict) -> int:
    draw.rectangle([(0, y), (PANEL_W, y + 145)], fill=BG_SECT)
    y = _sect_header(draw, y, '  실내 환경')
    y = _row2(draw, y,
              '온도',  f'{hvac.indoor_temp:.1f}°C', _temp_color(hvac.indoor_temp, False),
              '습도',  f'{hvac.indoor_humid:.0f}%', C_VAL)
    pmv = ds.get('pmv_val', 0.0)
    y = _row2(draw, y,
              'PMV',   f'{pmv:+.2f}',                    _pmv_color(pmv),
              '상태',  ds.get('comfort_msg', '-')[:14],   _pmv_color(pmv))
    khai_s, khai_c = _khai_str(ds.get('khai', 0))
    y = _row1(draw, y, '대기질', khai_s, khai_c)
    return y


def _draw_hvac(draw: ImageDraw.Draw, y: int, hvac, sm,
               manual_ctrl: dict = None) -> int:
    is_manual = manual_ctrl is not None and manual_ctrl.get("enabled", False)
    bg_col = (50, 30, 30) if is_manual else BG_SECT
    draw.rectangle([(0, y), (PANEL_W, y + 180)], fill=bg_col)

    # 섹션 헤더 — 수동 모드 시 강조 표시
    if is_manual:
        draw.rectangle([(0, y), (PANEL_W, y + 40)], fill=(130, 40, 40))
        draw.text((14, y + 8), '  에어컨 상태  ◀ 수동 조작 중',
                  font=_font(19, bold=True), fill=(255, 120, 120))
        y += 40
    else:
        y = _sect_header(draw, y, '  에어컨 상태  [M키: 수동 전환]')

    # 모드 색상 및 텍스트
    mode_col  = C_HEAT if hvac.mode == 'heat' else C_COOL
    mode_str  = f"{'난방' if hvac.mode == 'heat' else '냉방'}  {'ON' if hvac.is_on else 'OFF'}"
    y = _row2(draw, y,
              '모드',    mode_str,                   mode_col,
              '설정온도', f'{hvac.target_temp:.0f}°C', C_VAL)
    y = _row2(draw, y,
              '풍량',    f'Fan {hvac.fan_speed}',    C_VAL,
              '창문',    '열림' if hvac.window_open else '닫힘', C_CYAN)
    _state_labels = {'EMPTY': '공실', 'ARRIVAL': '도착',
                     'STEADY': '재실 중', 'LUNCH_BREAK': '점심 외출',
                     'PRE_DEPARTURE': '퇴실 준비'}
    occ_str = _state_labels.get(sm.state.value, sm.state.value)
    occ_col = (C_LABEL  if sm.state.value == 'EMPTY' else
               C_CYAN   if sm.state.value == 'LUNCH_BREAK' else C_GREEN)
    y = _row1(draw, y, '재실', occ_str, occ_col)

    # 수동 모드 조작 안내
    if is_manual:
        hint = 'P:전원  C:냉방  H:난방  +/-:온도  F:팬'
        draw.text((14, y + 4), hint, font=_font(16), fill=(200, 120, 120))
        y += 28

    return y


def _draw_occupancy(draw: ImageDraw.Draw, y: int, ds: dict) -> int:
    draw.rectangle([(0, y), (PANEL_W, y + 270)], fill=BG_SECT)
    y = _sect_header(draw, y, '  재실 / VLM 분석')

    people    = ds.get('people_count', 0)
    count_src = ds.get('count_source', 'YOLO').upper()
    p_col     = C_GREEN if people > 0 else C_LABEL
    src_col   = C_CYAN if count_src == 'YOLO' else C_ORANGE
    y = _row2(draw, y,
              '인원',   f'{people}명',                        p_col,
              '감지',   count_src,                            src_col)
    y = _row2(draw, y,
              '모션',   f"{ds.get('motion_score', 0.0):.1f}", C_VAL,
              'MET',   f"{ds.get('met', 1.0):.1f} ({ds.get('met_source','vlm').upper()})", C_VAL)
    y = _row1(draw, y, '활동', ds.get('activity', '-'), C_VAL)

    # 구분선
    draw.line([(8, y + 4), (PANEL_W - 8, y + 4)], fill=(55, 55, 75), width=1)
    y += 10

    clo       = ds.get('clo', 1.0)
    room_sz   = ds.get('room_size', 'medium')
    room_m2   = ds.get('room_size_m2', 30.0)
    outerwear = ds.get('outerwear', 'no')
    heat_src  = ds.get('heat_source', 'no')
    y = _row2(draw, y,
              'CLO',    f'{clo:.2f} clo',                     C_VAL,
              '방 크기', f'{room_sz} ({room_m2:.0f}㎡)',        C_VAL)
    ow_col   = C_ORANGE if outerwear == 'yes' else C_LABEL
    hs_col   = C_RED    if heat_src  == 'yes' else C_LABEL
    y = _row2(draw, y,
              '아우터',  '착용' if outerwear == 'yes' else '없음', ow_col,
              '열원',    '감지' if heat_src  == 'yes' else '없음', hs_col)
    y = _row1(draw, y, '분석', ds.get('last_analysis', '--:--:--'), C_TIME)
    return y


def _draw_solution(draw: ImageDraw.Draw, y: int, end_y: int,
                   state: SystemState, pmv: float, hvac, out_temp: float,
                   people: int) -> int:
    draw.rectangle([(0, y), (PANEL_W, end_y)], fill=(28, 28, 42))
    y = _sect_header(draw, y, '  솔루션')

    lines = _get_solution(state, pmv, hvac, out_temp, people)
    for line in lines:
        draw.text((14, y + 5), f'▶  {line}', font=_font(18), fill=C_GOLD)
        y += 34
    return end_y


def _draw_vlm_context(draw: ImageDraw.Draw, y: int,
                      vlm_data: dict | None, vlm_time: str | None,
                      analyzing: bool) -> int:
    """VLM 분석 결과 패널 — 대시보드 하단에 통합"""
    draw.rectangle([(0, y), (PANEL_W, y + 240)], fill=(18, 18, 32))
    draw.rectangle([(0, y), (PANEL_W, y + 40)], fill=(28, 24, 55))

    # 헤더
    status_col = C_ORANGE if analyzing else (C_GREEN if vlm_data else C_LABEL)
    status_txt = '분석중...' if analyzing else ('완료' if vlm_data else '대기중')
    draw.text((14, y + 10), '  VLM 분석', font=_font(19, bold=True), fill=C_TITLE)
    draw.text((PANEL_W - 90, y + 12), status_txt, font=_font(15), fill=status_col)
    if vlm_time:
        draw.text((200, y + 14), vlm_time, font=_font(13), fill=C_LABEL)
    y += 40

    if not vlm_data:
        draw.text((14, y + 16), 'VLM 분석 대기 중 — 카메라 프레임 수집 중...',
                  font=_font(15), fill=C_LABEL)
        return y + 200

    # Raw 출력
    draw.text((14, y + 6), 'Raw Output', font=_font(14, True), fill=(100, 180, 255))
    y += 26
    raw = str(vlm_data.get('raw_response', ''))
    max_w = 72
    for line in raw.replace('{', '{ ').replace(',', ', ').split('\n')[:3]:
        line = line.strip()
        if len(line) > max_w:
            line = line[:max_w] + '...'
        if line:
            draw.text((14, y), line, font=_font(13), fill=(185, 185, 160))
            y += 16

    # 구분선
    y += 4
    draw.line([(12, y), (PANEL_W - 12, y)], fill=(45, 42, 72), width=1)
    y += 8

    # 파싱 결과 (2열)
    draw.text((14, y), 'Parsed', font=_font(14, True), fill=(100, 180, 255))
    y += 22
    fields = [
        ('activity',    vlm_data.get('activity', '-')),
        ('clo',         f"{vlm_data.get('clo', 0):.2f} clo"),
        ('met',         f"{vlm_data.get('met', 0):.1f} met"),
        ('outerwear',   vlm_data.get('outerwear', '-')),
        ('room_size',   vlm_data.get('room_size', '-')),
        ('heat_source', vlm_data.get('heat_source', '-')),
    ]
    for i, (k, v) in enumerate(fields):
        col = 0 if i % 2 == 0 else PANEL_W // 2
        oc  = (C_ORANGE if v == 'yes' else
               C_RED    if k == 'heat_source' and v == 'yes' else C_VAL)
        draw.text((col + 14, y), f'{k:<12}', font=_font(13), fill=C_LABEL)
        draw.text((col + 110, y), v, font=_font(13, True), fill=oc)
        if i % 2 == 1:
            y += 18
    return y + 24


# ── 공개 API ──────────────────────────────────────────────────────────────────

def _draw_env_override(draw: ImageDraw.Draw, y: int,
                       env: dict, env_vars: list, env_label: dict) -> int:
    """환경 오버라이드 활성 시 표시되는 섹션"""
    draw.rectangle([(0, y), (PANEL_W, y + 92)], fill=(30, 20, 45))
    draw.rectangle([(0, y), (PANEL_W, y + 26)], fill=(110, 40, 110))
    draw.text((14, y + 8), '  ENV OVERRIDE  [E:OFF  [:prev  ]:next  +/-:조정]',
              font=_font(16, bold=True), fill=(220, 140, 255))
    y += 40
    sel = env_vars[env.get("selected", 0)]
    for var in env_vars:
        lbl  = env_label[var]
        val  = env.get(var, 0.0)
        unit = "%" if "humid" in var else "°C"
        is_sel = (var == sel)
        col  = (255, 220, 80) if is_sel else C_VAL
        prefix = "▶ " if is_sel else "  "
        draw.text((14, y + 4), f"{prefix}{lbl}", font=_font(17), fill=col)
        draw.text((180, y + 4), f"{val}{unit}", font=_font(18, bold=is_sel), fill=col)
        y += 26
    return y + 8


def build(cam_h: int, hvac, sm,
          out_temp: float, out_humid: float,
          out_weather: str, out_wind: float,
          ds: dict, manual_ctrl: dict = None,
          env_override: dict = None,
          vlm_data: dict = None,
          vlm_time: str = None,
          vlm_analyzing: bool = False) -> np.ndarray:
    """
    대시보드 패널 생성

    Args:
        cam_h       : 카메라 프레임 높이 (패널 높이에 맞춤)
        hvac        : HVACSimulator 인스턴스
        sm          : StateManager 인스턴스
        em          : EnergyMonitor 인스턴스
        out_temp    : 외부 기온 (°C)
        out_humid   : 외부 습도 (%)
        out_weather : 날씨 설명
        out_wind    : 풍속 (m/s)
        ds          : display_state dict (pmv_val, comfort_msg, people_count, ...)

    Returns:
        np.ndarray: BGR 이미지 (cam_h, PANEL_W, 3)
    """
    panel_h = max(cam_h, 960)
    img  = Image.new('RGB', (PANEL_W, panel_h), BG)
    draw = ImageDraw.Draw(img)

    # ── 구분선 (세로) ─────────────────────────────────────────────────────────
    draw.line([(0, 0), (0, cam_h)], fill=(70, 60, 100), width=2)

    _ENV_VARS  = ["indoor_temp", "outdoor_temp", "indoor_humid", "outdoor_humid"]
    _ENV_LABEL = {"indoor_temp": "실내온도", "outdoor_temp": "실외온도",
                  "indoor_humid": "실내습도", "outdoor_humid": "실외습도"}

    y  = 0
    y  = _draw_header(draw, y)
    if env_override and env_override.get("enabled"):
        y = _draw_env_override(draw, y, env_override, _ENV_VARS, _ENV_LABEL)
    y  = _draw_outdoor(draw, y, out_temp, out_humid, out_weather, out_wind, ds)
    y  = _draw_indoor(draw, y, hvac, ds)
    y  = _draw_hvac(draw, y, hvac, sm, manual_ctrl)
    y  = _draw_occupancy(draw, y, ds)

    # VLM 컨텍스트 패널 (별도 창 대신 패널 하단에 통합)
    vlm_end = y + 240
    if vlm_end <= panel_h:
        y = _draw_vlm_context(draw, y, vlm_data, vlm_time, vlm_analyzing)

    solution_y = y
    solution_e = panel_h
    if solution_e > solution_y + 30:
        _draw_solution(draw, solution_y, solution_e,
                       sm.state, ds.get('pmv_val', 0.0),
                       hvac, out_temp, ds.get('people_count', 0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
