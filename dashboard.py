"""
[대시보드 UI 모듈 — 라이트 테마]
카메라 화면 옆에 표시될 정보 패널을 PIL로 렌더링합니다.
밝은 흰색 카드 스타일 — 흰 카드 + 옅은 테두리 + 포인트 컬러.
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
PAD     = 14          # 패널 좌우 여백
GAP     = 8           # 카드 간 간격
ROW_H   = 34          # 정보 행 높이

# ── 색상 팔레트 (RGB, 라이트) ─────────────────────────────────────────────────
BG        = (244, 246, 249)   # 전체 배경 (아주 옅은 회색)
CARD      = (255, 255, 255)   # 카드 배경
BORDER    = (228, 231, 238)   # 카드 테두리
C_TXT     = ( 32,  36,  48)   # 본문 값 (진회색)
C_LABEL   = (138, 144, 160)   # 라벨 (중간 회색)
C_TITLE   = ( 70,  78, 100)   # 섹션 타이틀
C_MUTED   = (170, 175, 188)   # 흐린 텍스트 (타임스탬프 등)

C_ACCENT  = ( 79, 108, 255)   # 포인트 블루
C_GREEN   = ( 22, 163,  74)   # 쾌적/정상
C_ORANGE  = (228, 138,  10)   # 경고
C_RED     = (220,  56,  56)   # 위험
C_HEAT    = (234, 108,  30)   # 난방
C_COOL    = ( 37, 118, 235)   # 냉방
C_TEAL    = ( 13, 148, 136)   # 보조 정보

# 솔루션 카드 (옅은 앰버)
SOL_BG     = (255, 251, 235)
SOL_BORDER = (250, 229, 160)
SOL_TXT    = (146, 100,  10)

# 수동 모드 카드 (옅은 레드)
MAN_BG     = (254, 242, 242)
MAN_BORDER = (250, 180, 180)
MAN_TXT    = (185,  48,  48)

# 환경 오버라이드 카드 (옅은 퍼플)
ENV_BG     = (246, 243, 255)
ENV_BORDER = (210, 198, 250)
ENV_TXT    = (108,  82, 210)

# Raw 출력 코드 블록
CODE_BG    = (246, 247, 250)
CODE_TXT   = (100, 108, 128)

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


# ── 드로잉 헬퍼 ───────────────────────────────────────────────────────────────

def _rrect(draw, x0, y0, x1, y1, r, fill, outline=None, width=1):
    try:
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=r,
                               fill=fill, outline=outline, width=width)
    except AttributeError:  # 구버전 PIL
        draw.rectangle([(x0, y0), (x1, y1)], fill=fill, outline=outline, width=width)


def _card(draw, y: int, h: int, bg=CARD, border=BORDER) -> int:
    """둥근 카드 배경. 카드 내부 시작 y 반환."""
    _rrect(draw, PAD, y, PANEL_W - PAD, y + h, 12, bg, border)
    return y


def _card_title(draw, y: int, title: str, color=C_TITLE,
                right_text: str = None, right_color=C_MUTED) -> int:
    """카드 제목 (좌측 포인트 바 + 텍스트). 다음 행 y 반환."""
    bar_col = color if color != C_TITLE else C_ACCENT
    _rrect(draw, PAD + 14, y + 9, PAD + 18, y + 27, 2, bar_col)
    draw.text((PAD + 26, y + 6), title, font=_font(18, bold=True), fill=color)
    if right_text:
        f  = _font(14)
        tw = f.getbbox(right_text)[2] - f.getbbox(right_text)[0]
        draw.text((PANEL_W - PAD - 14 - tw, y + 10), right_text,
                  font=f, fill=right_color)
    return y + 34


def _row2(draw, y: int,
          lbl1: str, val1: str, col1: tuple,
          lbl2: str, val2: str, col2: tuple) -> int:
    x1, x2 = PAD + 26, PAD + 340
    draw.text((x1,       y + 5), lbl1, font=_font(17), fill=C_LABEL)
    draw.text((x1 + 96,  y + 3), val1, font=_font(20, bold=True), fill=col1)
    draw.text((x2,       y + 5), lbl2, font=_font(17), fill=C_LABEL)
    draw.text((x2 + 96,  y + 3), val2, font=_font(20, bold=True), fill=col2)
    return y + ROW_H


def _row1(draw, y: int, lbl: str, val: str, col: tuple = None) -> int:
    x1 = PAD + 26
    draw.text((x1,      y + 5), lbl, font=_font(17), fill=C_LABEL)
    draw.text((x1 + 96, y + 3), val, font=_font(20), fill=col or C_TXT)
    return y + ROW_H


def _divider(draw, y: int) -> int:
    draw.line([(PAD + 16, y + 3), (PANEL_W - PAD - 16, y + 3)],
              fill=(238, 240, 245), width=1)
    return y + 8


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
        if temp < 15:  return C_TXT
        if temp < 28:  return C_GREEN
        return C_RED
    else:
        if temp < 18:  return C_COOL
        if temp < 27:  return C_GREEN
        return C_RED


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

    if state == SystemState.LUNCH_BREAK:
        return ["점심 외출 감지", "복귀 대비 약운전 유지 중"]

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

def _draw_header(draw, y: int) -> int:
    """상단 헤더 — 흰 카드, 타이틀 + 시각"""
    h = 58
    _card(draw, y, h)
    # 포인트 도트
    _rrect(draw, PAD + 16, y + 22, PAD + 30, y + 36, 7, C_ACCENT)
    draw.text((PAD + 42, y + 13), 'VLM HVAC SYSTEM',
              font=_font(24, bold=True), fill=C_TXT)
    ts = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    f  = _font(15)
    tw = f.getbbox(ts)[2] - f.getbbox(ts)[0]
    draw.text((PANEL_W - PAD - 16 - tw, y + 20), ts, font=f, fill=C_MUTED)
    return y + h + GAP


def _draw_outdoor(draw, y: int, temp: float, humid: float,
                  weather: str, wind: float) -> int:
    h = 34 + ROW_H * 2 + 8
    _card(draw, y, h)
    cy = _card_title(draw, y, '실외 환경')
    cy = _row2(draw, cy,
               '기온', f'{temp:.1f}°C', _temp_color(temp, True),
               '습도', f'{humid:.0f}%', C_TXT)
    cy = _row2(draw, cy,
               '날씨', weather[:12], C_TEAL,
               '풍속', f'{wind:.1f} m/s', C_TXT)
    return y + h + GAP


def _draw_indoor(draw, y: int, hvac, ds: dict) -> int:
    h = 34 + ROW_H * 2 + 8
    _card(draw, y, h)
    cy = _card_title(draw, y, '실내 환경')
    cy = _row2(draw, cy,
               '온도', f'{hvac.indoor_temp:.1f}°C', _temp_color(hvac.indoor_temp, False),
               '습도', f'{hvac.indoor_humid:.0f}%', C_TXT)
    pmv = ds.get('pmv_val', 0.0)
    comfort = ds.get('comfort_msg', '-').split(' (')[0]   # 영문 병기 제거
    cy = _row2(draw, cy,
               'PMV',  f'{pmv:+.2f}', _pmv_color(pmv),
               '상태', comfort,        _pmv_color(pmv))
    return y + h + GAP


def _draw_hvac(draw, y: int, hvac, sm, manual_ctrl: dict = None) -> int:
    is_manual = manual_ctrl is not None and manual_ctrl.get("enabled", False)
    h = 34 + ROW_H * 2 + 8 + (28 if is_manual else 0)

    if is_manual:
        _card(draw, y, h, bg=MAN_BG, border=MAN_BORDER)
        cy = _card_title(draw, y, '에어컨 상태 · 수동 조작 중', color=MAN_TXT)
    else:
        _card(draw, y, h)
        cy = _card_title(draw, y, '에어컨 상태', right_text='M키: 수동 전환')

    mode_col = C_HEAT if hvac.mode == 'heat' else C_COOL
    mode_str = f"{'난방' if hvac.mode == 'heat' else '냉방'} {'ON' if hvac.is_on else 'OFF'}"
    if not hvac.is_on:
        mode_col = C_LABEL
    cy = _row2(draw, cy,
               '모드',     mode_str,                    mode_col,
               '설정온도', f'{hvac.target_temp:.0f}°C', C_TXT)

    _state_labels = {'EMPTY': '공실', 'ARRIVAL': '도착',
                     'STEADY': '재실 중', 'LUNCH_BREAK': '점심 외출',
                     'PRE_DEPARTURE': '퇴실 준비'}
    occ_str = _state_labels.get(sm.state.value, sm.state.value)
    occ_col = (C_LABEL  if sm.state.value == 'EMPTY' else
               C_TEAL   if sm.state.value == 'LUNCH_BREAK' else
               C_ORANGE if sm.state.value == 'PRE_DEPARTURE' else C_GREEN)
    cy = _row2(draw, cy,
               '풍량', f'Fan {hvac.fan_speed}', C_TXT,
               '재실', occ_str,                 occ_col)

    if is_manual:
        draw.text((PAD + 26, cy + 2), 'P:전원  C:냉방  H:난방  +/-:온도  F:팬',
                  font=_font(15), fill=MAN_TXT)

    return y + h + GAP


def _draw_occupancy(draw, y: int, ds: dict) -> int:
    h = 34 + ROW_H * 5 + 8 + 10
    _card(draw, y, h)
    cy = _card_title(draw, y, '재실 / VLM 분석',
                     right_text=ds.get('last_analysis', '--:--:--'))

    people    = ds.get('people_count', 0)
    count_src = ds.get('count_source', 'YOLO').upper()
    p_col     = C_GREEN if people > 0 else C_LABEL
    src_col   = C_TEAL if count_src == 'YOLO' else C_ORANGE
    cy = _row2(draw, cy,
               '인원', f'{people}명', p_col,
               '감지', count_src,     src_col)
    cy = _row2(draw, cy,
               '모션', f"{ds.get('motion_score', 0.0):.1f}", C_TXT,
               'MET',  f"{ds.get('met', 1.0):.1f} ({ds.get('met_source','vlm').upper()})", C_TXT)
    cy = _row1(draw, cy, '활동', ds.get('activity', '-'), C_TXT)

    cy = _divider(draw, cy)

    clo       = ds.get('clo', 1.0)
    room_sz   = ds.get('room_size', 'medium')
    room_m2   = ds.get('room_size_m2', 30.0)
    outerwear = ds.get('outerwear', 'no')
    heat_src  = ds.get('heat_source', 'no')
    cy = _row2(draw, cy,
               'CLO',    f'{clo:.2f} clo',              C_TXT,
               '방 크기', f'{room_sz} ({room_m2:.0f}㎡)', C_TXT)
    ow_col = C_ORANGE if outerwear == 'yes' else C_LABEL
    hs_col = C_RED    if heat_src  == 'yes' else C_LABEL
    cy = _row2(draw, cy,
               '아우터', '착용' if outerwear == 'yes' else '없음', ow_col,
               '열원',   '감지' if heat_src  == 'yes' else '없음', hs_col)
    return y + h + GAP


def _draw_vlm_context(draw, y: int, vlm_data: dict | None,
                      vlm_time: str | None, analyzing: bool) -> int:
    """VLM 분석 결과 카드 — Raw 출력 + 파싱 결과"""
    h = 34 + 76 + 8 + 23 * 3 + 12
    _card(draw, y, h)

    status_col = C_ORANGE if analyzing else (C_GREEN if vlm_data else C_LABEL)
    status_txt = '분석중...' if analyzing else ('완료' if vlm_data else '대기중')
    cy = _card_title(draw, y, 'VLM 컨텍스트',
                     right_text=status_txt, right_color=status_col)

    if not vlm_data:
        draw.text((PAD + 26, cy + 12), 'VLM 분석 대기 중 — 카메라 프레임 수집 중...',
                  font=_font(16), fill=C_LABEL)
        return y + h + GAP

    # Raw 출력 (코드 블록) — 긴 한 줄은 자동 줄바꿈해 최대 3줄 표시
    _rrect(draw, PAD + 16, cy, PANEL_W - PAD - 16, cy + 76, 8, CODE_BG)
    import textwrap
    raw   = ' '.join(str(vlm_data.get('raw_response', '')).split())
    lines = textwrap.wrap(raw, width=60)[:3]
    if len(lines) == 3 and len(raw) > 180:
        lines[2] = lines[2][:57] + '...'
    ty = cy + 8
    for line in lines:
        draw.text((PAD + 26, ty), line, font=_font(14), fill=CODE_TXT)
        ty += 20
    cy += 76 + 8

    # 파싱 결과 (2열)
    fields = [
        ('activity',    vlm_data.get('activity', '-')),
        ('clo',         f"{vlm_data.get('clo', 0):.2f} clo"),
        ('met',         f"{vlm_data.get('met', 0):.1f} met"),
        ('outerwear',   vlm_data.get('outerwear', '-')),
        ('room_size',   vlm_data.get('room_size', '-')),
        ('heat_source', vlm_data.get('heat_source', '-')),
    ]
    half = (PANEL_W - PAD * 2) // 2
    for i, (k, v) in enumerate(fields):
        col_x = PAD + 26 + (0 if i % 2 == 0 else half)
        v_col = (C_RED    if k == 'heat_source' and v == 'yes' else
                 C_ORANGE if v == 'yes' else C_TXT)
        draw.text((col_x,       cy), k,      font=_font(15), fill=C_LABEL)
        draw.text((col_x + 110, cy), str(v), font=_font(15, bold=True), fill=v_col)
        if i % 2 == 1:
            cy += 23
    return y + h + GAP


def _draw_solution(draw, y: int, end_y: int,
                   state: SystemState, pmv: float, hvac, out_temp: float,
                   people: int) -> int:
    """하단 솔루션 카드 (옅은 앰버)"""
    if end_y - y < 60:
        return end_y
    _rrect(draw, PAD, y, PANEL_W - PAD, end_y - GAP, 12, SOL_BG, SOL_BORDER)
    cy = _card_title(draw, y, '솔루션', color=SOL_TXT)
    for line in _get_solution(state, pmv, hvac, out_temp, people):
        draw.text((PAD + 26, cy + 3), f'·  {line}',
                  font=_font(18, bold=True), fill=SOL_TXT)
        cy += 32
    return end_y


def _draw_env_override(draw, y: int, env: dict,
                       env_vars: list, env_label: dict) -> int:
    """환경 오버라이드 활성 시 표시되는 카드 (옅은 퍼플)"""
    h = 34 + 26 * len(env_vars) + 8
    _card(draw, y, h, bg=ENV_BG, border=ENV_BORDER)
    cy = _card_title(draw, y, 'ENV OVERRIDE', color=ENV_TXT,
                     right_text='E:OFF  [ ]:선택  +/-:조정', right_color=ENV_TXT)
    sel = env_vars[env.get("selected", 0)]
    for var in env_vars:
        lbl    = env_label[var]
        val    = env.get(var, 0.0)
        unit   = "%" if "humid" in var else "°C"
        is_sel = (var == sel)
        col    = ENV_TXT if is_sel else C_LABEL
        prefix = "▶ " if is_sel else "    "
        draw.text((PAD + 26, cy + 2), f"{prefix}{lbl}",
                  font=_font(16, bold=is_sel), fill=col)
        draw.text((PAD + 200, cy + 2), f"{val}{unit}",
                  font=_font(16, bold=is_sel), fill=col)
        cy += 26
    return y + h + GAP


# ── 공개 API ──────────────────────────────────────────────────────────────────

def build(cam_h: int, hvac, sm,
          out_temp: float, out_humid: float,
          out_weather: str, out_wind: float,
          ds: dict, manual_ctrl: dict = None,
          env_override: dict = None,
          vlm_data: dict = None,
          vlm_time: str = None,
          vlm_analyzing: bool = False) -> np.ndarray:
    """
    대시보드 패널 생성 (라이트 테마)

    Args:
        cam_h       : 카메라 프레임 높이 (패널 높이에 맞춤)
        hvac        : HVACSimulator 인스턴스
        sm          : StateManager 인스턴스
        out_temp    : 외부 기온 (°C)
        out_humid   : 외부 습도 (%)
        out_weather : 날씨 설명
        out_wind    : 풍속 (m/s)
        ds          : display_state dict (pmv_val, comfort_msg, ai_wh, ...)

    Returns:
        np.ndarray: BGR 이미지 (panel_h, PANEL_W, 3)
    """
    panel_h = max(cam_h, 1100)
    img  = Image.new('RGB', (PANEL_W, panel_h), BG)
    draw = ImageDraw.Draw(img)

    _ENV_VARS  = ["indoor_temp", "outdoor_temp", "indoor_humid", "outdoor_humid"]
    _ENV_LABEL = {"indoor_temp": "실내온도", "outdoor_temp": "실외온도",
                  "indoor_humid": "실내습도", "outdoor_humid": "실외습도"}

    y = GAP
    y = _draw_header(draw, y)
    if env_override and env_override.get("enabled"):
        y = _draw_env_override(draw, y, env_override, _ENV_VARS, _ENV_LABEL)
    y = _draw_outdoor(draw, y, out_temp, out_humid, out_weather, out_wind)
    y = _draw_indoor(draw, y, hvac, ds)
    y = _draw_hvac(draw, y, hvac, sm, manual_ctrl)
    y = _draw_occupancy(draw, y, ds)

    # VLM 컨텍스트 카드 — 솔루션 최소 공간(92px)이 남을 때만
    if y + 207 <= panel_h - 92:
        y = _draw_vlm_context(draw, y, vlm_data, vlm_time, vlm_analyzing)

    _draw_solution(draw, y, panel_h,
                   sm.state, ds.get('pmv_val', 0.0),
                   hvac, out_temp, ds.get('people_count', 0))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
