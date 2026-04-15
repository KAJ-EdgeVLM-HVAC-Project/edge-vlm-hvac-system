"""
[사용자 인터페이스 패널 — 삼성 시스템에어컨 스타일]
흰색 배경, 카드 레이아웃, 깔끔한 타이포그래피.
운영자 대시보드와 별도 창으로 표시.
"""

import cv2
import numpy as np
import platform
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 500
PANEL_H = 820

# ── 색상 (RGB) ─────────────────────────────────────────────────────────────────
BG           = (245, 247, 252)   # 연한 회색 배경
CARD         = (255, 255, 255)   # 흰 카드
HEADER_BG    = ( 18,  24,  56)   # 삼성 다크 네이비
HEADER_TXT   = (255, 255, 255)
SECTION_BG   = (237, 240, 248)   # 섹션 구분 배경
BORDER       = (218, 220, 230)

TXT_MAIN     = ( 22,  22,  36)   # 주 텍스트
TXT_SUB      = ( 98, 100, 118)   # 보조 텍스트
TXT_HINT     = (155, 158, 175)   # 힌트

COOL_BLU     = ( 30, 120, 215)   # 냉방 블루
HEAT_ORG     = (220,  75,  35)   # 난방 오렌지
GOOD_GRN     = ( 34, 170,  90)   # 쾌적 / 좋음
WARN_YLW     = (230, 155,  20)   # 경고
DANG_RED     = (200,  50,  50)   # 위험
VGOOD_TEAL   = ( 22, 190, 155)   # 아주좋음
PURPLE       = (100,  90, 200)   # 예측 등

# ── 폰트 캐시 ──────────────────────────────────────────────────────────────────
_fc: dict = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _fc:
        return _fc[key]
    sys = platform.system()
    if sys == 'Windows':
        root = os.environ.get('SystemRoot', r'C:\Windows')
        cands = [os.path.join(root, 'Fonts', 'malgunbd.ttf' if bold else 'malgun.ttf'),
                 os.path.join(root, 'Fonts', 'malgun.ttf')]
    elif sys == 'Darwin':
        cands = ['/System/Library/Fonts/AppleSDGothicNeo.ttc',
                 '/System/Library/Fonts/Supplemental/AppleGothic.ttf']
    else:
        cands = ['/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf' if bold
                 else '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for p in cands:
        try:
            f = ImageFont.truetype(p, size)
            _fc[key] = f
            return f
        except Exception:
            pass
    f = ImageFont.load_default()
    _fc[key] = f
    return f


# ── 드로잉 헬퍼 ───────────────────────────────────────────────────────────────

def _card(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int,
          fill=CARD, radius: int = 12, border_col=BORDER):
    """흰색 둥근 카드"""
    try:
        draw.rounded_rectangle([(x, y), (x + w, y + h)],
                                radius=radius, fill=fill, outline=border_col, width=1)
    except AttributeError:
        # Pillow < 8.2 fallback
        draw.rectangle([(x, y), (x + w, y + h)], fill=fill, outline=border_col, width=1)


def _badge(draw: ImageDraw.Draw, x: int, y: int, text: str,
           bg: tuple, txt: tuple = (255, 255, 255), radius: int = 8):
    """컬러 뱃지"""
    fw = _font(14, bold=True)
    bb = fw.getbbox(text)
    tw = bb[2] - bb[0]
    bw = tw + 18
    bh = 24
    try:
        draw.rounded_rectangle([(x, y), (x + bw, y + bh)],
                                radius=radius, fill=bg)
    except AttributeError:
        draw.rectangle([(x, y), (x + bw, y + bh)], fill=bg)
    draw.text((x + 9, y + 5), text, font=fw, fill=txt)
    return bw


def _pmv_bar(draw: ImageDraw.Draw, x: int, y: int, pmv: float) -> None:
    """PMV 5단계 컬러 바 (삼성 스타일)"""
    levels = [
        ('매우추움', (100, 160, 230)),
        ('추움',    (160, 200, 240)),
        ('쾌적',    ( 60, 190, 110)),
        ('더움',    (240, 160,  50)),
        ('매우더움', (220,  60,  60)),
    ]
    seg_w = 56
    seg_h = 10
    gap   = 4

    if pmv < -1.5:   active = 0
    elif pmv < -0.5: active = 1
    elif pmv <= 0.5: active = 2
    elif pmv <= 1.5: active = 3
    else:            active = 4

    cx = x
    for i, (lbl, col) in enumerate(levels):
        alpha_col = col if i == active else tuple(int(c * 0.28 + 220 * 0.72) for c in col)
        try:
            draw.rounded_rectangle(
                [(cx, y), (cx + seg_w, y + seg_h)],
                radius=4, fill=alpha_col)
        except AttributeError:
            draw.rectangle([(cx, y), (cx + seg_w, y + seg_h)], fill=alpha_col)
        if i == active:
            # 위에 레이블
            lf = _font(11, bold=True)
            lb = lf.getbbox(lbl)
            lw = lb[2] - lb[0]
            draw.text((cx + (seg_w - lw) // 2, y - 16), lbl, font=lf, fill=col)
        cx += seg_w + gap


def _fan_dots(draw: ImageDraw.Draw, x: int, y: int, speed: int) -> None:
    """팬 속도 점 3개"""
    for i in range(3):
        filled = i < speed
        col = COOL_BLU if filled else BORDER
        r = 7
        cx = x + i * 22
        draw.ellipse([(cx, y), (cx + r * 2, y + r * 2)], fill=col)


# ── 공기질 헬퍼 ───────────────────────────────────────────────────────────────

def _aq_level(pm10: int):
    if pm10 <= 15:  return '아주좋음', VGOOD_TEAL
    if pm10 <= 30:  return '좋음',     GOOD_GRN
    if pm10 <= 80:  return '보통',     WARN_YLW
    if pm10 <= 150: return '나쁨',     HEAT_ORG
    return '매우나쁨', DANG_RED


def _pm25_level(pm25: int):
    if pm25 <= 8:   return '아주좋음', VGOOD_TEAL
    if pm25 <= 15:  return '좋음',     GOOD_GRN
    if pm25 <= 35:  return '보통',     WARN_YLW
    if pm25 <= 75:  return '나쁨',     HEAT_ORG
    return '매우나쁨', DANG_RED


# ── 창문 권장 ─────────────────────────────────────────────────────────────────

def _window_msg(hvac, out_temp: float, pm10: int, pmv: float, people: int):
    if people == 0:
        return '공실 — 창문 상태 유지', TXT_SUB
    if pm10 > 150:
        return f'미세먼지 매우나쁨 — 창문 즉시 닫으세요', DANG_RED
    if pm10 > 80:
        return f'미세먼지 나쁨 ({pm10}㎍) — 창문 닫으세요', HEAT_ORG
    if hvac.is_on:
        m = '난방' if hvac.mode == 'heat' else '냉방'
        return f'{m} 가동 중 — 창문 닫으세요', WARN_YLW
    if pmv > 1.0 and out_temp < hvac.indoor_temp - 2.0:
        return f'실외 시원 ({out_temp:.1f}°C) — 창문 열어 환기', COOL_BLU
    if abs(pmv) <= 0.5 and pm10 <= 30:
        return '공기 맑음 — 창문 환기 권장', GOOD_GRN
    return f'실내 {hvac.indoor_temp:.1f}°C — 창문 현재 상태 유지', TXT_SUB


# ── 공개 API ──────────────────────────────────────────────────────────────────

def build(hvac, sm, ds: dict, em_data: dict,
          pmv_preference: float, pmv_history: list,  # pmv_history: 호환성 유지
          occ_pred: dict, out_temp: float) -> np.ndarray:
    """
    삼성 시스템에어컨 스타일 사용자 UI 패널.
    pmv_history / occ_pred 는 서명 호환성 유지 (이 뷰에서는 미사용).
    """
    _ = pmv_history, occ_pred   # 미사용 파라미터 (호환성 유지)
    img  = Image.new('RGB', (PANEL_W, PANEL_H), BG)
    draw = ImageDraw.Draw(img)

    PAD  = 16   # 좌우 여백
    CW   = PANEL_W - PAD * 2   # 카드 너비

    # ════════════════════════════════════════════════════════════════
    # 1. 헤더
    # ════════════════════════════════════════════════════════════════
    draw.rectangle([(0, 0), (PANEL_W, 68)], fill=HEADER_BG)
    draw.text((PAD, 10), 'VLM HVAC',
              font=_font(22, bold=True), fill=HEADER_TXT)
    draw.text((PAD, 38), '스마트 에어컨 제어',
              font=_font(14), fill=(160, 170, 210))

    time_str = datetime.now().strftime('%H:%M')
    tf = _font(20, bold=True)
    tb = tf.getbbox(time_str)
    draw.text((PANEL_W - PAD - (tb[2] - tb[0]), 12), time_str,
              font=tf, fill=HEADER_TXT)

    state_map = {'EMPTY': ('공실', TXT_HINT),
                 'ARRIVAL': ('도착', (140, 200, 255)),
                 'STEADY': ('운전 중', (120, 230, 160)),
                 'PRE_DEPARTURE': ('절전', (255, 190, 80))}
    slbl, scol = state_map.get(sm.state.value, (sm.state.value, TXT_HINT))
    draw.text((PANEL_W - PAD - 52, 40), slbl, font=_font(13), fill=scol)

    y = 80

    # ════════════════════════════════════════════════════════════════
    # 2. 온도 메인 카드
    # ════════════════════════════════════════════════════════════════
    card_h = 198
    _card(draw, PAD, y, CW, card_h)

    indoor = hvac.indoor_temp
    pmv    = ds.get('pmv_val', 0.0)

    # 실내 온도 (초대형)
    t_col = (COOL_BLU if indoor < 20 else
             GOOD_GRN if indoor < 27 else HEAT_ORG)
    tf_big = _font(58, bold=True)
    t_str  = f'{indoor:.1f}°C'
    draw.text((PAD + 20, y + 18), t_str, font=tf_big, fill=t_col)

    # PMV 뱃지 (우측 상단)
    pmv_col = (GOOD_GRN if abs(pmv) <= 0.5 else
               WARN_YLW if abs(pmv) <= 1.5 else DANG_RED)
    comfort = ds.get('comfort_msg', '-')
    _badge(draw, PAD + CW - 90, y + 18, comfort[:6], pmv_col)

    # 실외 온도 (작게)
    draw.text((PAD + 20, y + 86), f'실외  {out_temp:.1f}°C',
              font=_font(14), fill=TXT_SUB)

    # PMV 바
    bar_x = PAD + 20
    bar_y = y + 112
    _pmv_bar(draw, bar_x, bar_y + 18, pmv)

    # 선호 설정
    if pmv_preference > 0:
        pref_txt = f'선호  따뜻하게  +{pmv_preference:.1f}'
        pref_col = HEAT_ORG
    elif pmv_preference < 0:
        pref_txt = f'선호  시원하게  {pmv_preference:.1f}'
        pref_col = COOL_BLU
    else:
        pref_txt = '선호  중립 (표준 쾌적도)'
        pref_col = TXT_SUB

    draw.text((PAD + 20, y + 156), pref_txt, font=_font(14), fill=pref_col)
    draw.text((PAD + CW - 100, y + 156), '▲U  ▼D',
              font=_font(13, bold=True), fill=TXT_HINT)

    y += card_h + 12

    # ════════════════════════════════════════════════════════════════
    # 3. 에어컨 상태 바
    # ════════════════════════════════════════════════════════════════
    bar_h = 60
    _card(draw, PAD, y, CW, bar_h, fill=SECTION_BG)

    # 모드 + ON/OFF
    if not hvac.is_on:
        mode_txt, mode_col = 'OFF', TXT_HINT
    elif hvac.mode == 'cool':
        mode_txt, mode_col = '냉방', COOL_BLU
    else:
        mode_txt, mode_col = '난방', HEAT_ORG
    draw.text((PAD + 18, y + 10), mode_txt,
              font=_font(20, bold=True), fill=mode_col)

    # 설정 온도
    draw.text((PAD + 90, y + 10), f'설정  {hvac.target_temp:.0f}°C',
              font=_font(16), fill=TXT_MAIN)

    # 팬 속도 도트
    draw.text((PAD + 240, y + 10), '팬', font=_font(14), fill=TXT_SUB)
    _fan_dots(draw, PAD + 272, y + 14, hvac.fan_speed if hvac.is_on else 0)

    # 창문
    win_txt = '창문 열림' if hvac.window_open else '창문 닫힘'
    win_col = COOL_BLU if hvac.window_open else TXT_HINT
    draw.text((PAD + 360, y + 10), win_txt, font=_font(14), fill=win_col)

    y += bar_h + 12

    # ════════════════════════════════════════════════════════════════
    # 4. 재실 + 인원
    # ════════════════════════════════════════════════════════════════
    occ_h = 50
    _card(draw, PAD, y, CW, occ_h, fill=CARD)

    people    = ds.get('people_count', 0)
    last_ana  = ds.get('last_analysis', '--:--:--')
    p_col     = GOOD_GRN if people > 0 else TXT_HINT
    draw.text((PAD + 18, y + 12), f'재실  {people}명',
              font=_font(17, bold=True), fill=p_col)
    draw.text((PAD + 130, y + 14), f'마지막 분석  {last_ana}',
              font=_font(13), fill=TXT_HINT)

    y += occ_h + 12

    # ════════════════════════════════════════════════════════════════
    # 5. 공기질 카드
    # ════════════════════════════════════════════════════════════════
    aq_h = 108
    _card(draw, PAD, y, CW, aq_h)

    draw.text((PAD + 18, y + 12), '공기질',
              font=_font(15, bold=True), fill=TXT_MAIN)

    pm10 = int(ds.get('pm10', 0))
    pm25 = int(ds.get('pm25', 0))
    pm10_lv, pm10_c = _aq_level(pm10)
    pm25_lv, pm25_c = _pm25_level(pm25)

    # PM10
    draw.text((PAD + 18,  y + 40), 'PM10',  font=_font(13), fill=TXT_SUB)
    draw.text((PAD + 18,  y + 58), f'{pm10} ㎍/㎥', font=_font(15), fill=TXT_MAIN)
    _badge(draw, PAD + 18, y + 80, pm10_lv, pm10_c)

    # PM2.5
    draw.text((PAD + 180, y + 40), 'PM2.5', font=_font(13), fill=TXT_SUB)
    draw.text((PAD + 180, y + 58), f'{pm25} ㎍/㎥', font=_font(15), fill=TXT_MAIN)
    _badge(draw, PAD + 180, y + 80, pm25_lv, pm25_c)

    y += aq_h + 12

    # ════════════════════════════════════════════════════════════════
    # 6. 창문 권장
    # ════════════════════════════════════════════════════════════════
    win_h = 52
    _card(draw, PAD, y, CW, win_h, fill=SECTION_BG)

    wm, wc = _window_msg(hvac, out_temp, pm10, pmv, people)
    draw.text((PAD + 18, y + 8),  '창문',  font=_font(13, bold=True), fill=TXT_SUB)
    draw.text((PAD + 18, y + 26), wm[:42], font=_font(14), fill=wc)

    y += win_h + 12

    # ════════════════════════════════════════════════════════════════
    # 7. 에너지 절약 카드
    # ════════════════════════════════════════════════════════════════
    eng_h = 100
    _card(draw, PAD, y, CW, eng_h)

    draw.text((PAD + 18, y + 12), '에너지 절약',
              font=_font(15, bold=True), fill=TXT_MAIN)

    saved_kwh   = em_data.get('saved_kwh',   0.0)
    savings_pct = em_data.get('savings_pct', 0.0)
    comfort_pct = em_data.get('comfort_pct', 0.0)
    co2_g       = int(max(0.0, saved_kwh) * 454.5)
    money_won   = int(max(0.0, saved_kwh) * 120)

    s_col = GOOD_GRN if saved_kwh >= 0 else DANG_RED

    # 행1: kWh + 절약률
    draw.text((PAD + 18,  y + 40), '절약량', font=_font(12), fill=TXT_SUB)
    draw.text((PAD + 18,  y + 56), f'{saved_kwh:.4f} kWh',
              font=_font(16, bold=True), fill=s_col)

    draw.text((PAD + 170, y + 40), '절약률', font=_font(12), fill=TXT_SUB)
    draw.text((PAD + 170, y + 56), f'{savings_pct:.1f}%',
              font=_font(16, bold=True), fill=s_col)

    draw.text((PAD + 290, y + 40), '쾌적유지율', font=_font(12), fill=TXT_SUB)
    draw.text((PAD + 290, y + 56), f'{comfort_pct:.1f}%',
              font=_font(16, bold=True),
              fill=GOOD_GRN if comfort_pct >= 70 else WARN_YLW)

    # 행2: CO2 + 금액
    draw.text((PAD + 18,  y + 78), f'CO₂ {co2_g}g 감축',
              font=_font(13), fill=TXT_HINT)
    draw.text((PAD + 200, y + 78), f'절약금액  {money_won:,}원',
              font=_font(13), fill=TXT_HINT)

    y += eng_h + 12

    # ════════════════════════════════════════════════════════════════
    # 8. 하단 키 힌트
    # ════════════════════════════════════════════════════════════════
    hint_y = PANEL_H - 36
    draw.line([(PAD, hint_y), (PANEL_W - PAD, hint_y)], fill=BORDER, width=1)
    draw.text((PAD, hint_y + 6),
              'U: 따뜻하게   D: 시원하게   S: VLM 분석   Q: 종료',
              font=_font(12), fill=TXT_HINT)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
