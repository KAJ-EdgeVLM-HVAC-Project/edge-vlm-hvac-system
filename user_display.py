"""
[사용자 인터페이스 — 모바일 앱 스타일]
마우스 클릭 버튼 기반 (키보드 불필요).
"추워요" / "더워요" 버튼으로 PMV 선호 조절.
"""

import cv2
import numpy as np
import platform
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 460
PANEL_H = 690

# ── 색상 ──────────────────────────────────────────────────────────────────────
BG          = (250, 251, 255)
CARD        = (255, 255, 255)
HEADER_BG   = ( 17,  24,  52)
HDR_TXT     = (255, 255, 255)
HDR_SUB     = (140, 150, 200)

TXT         = ( 20,  20,  36)
TXT_S       = ( 98, 102, 125)
TXT_H       = (170, 172, 190)
BORDER      = (220, 223, 235)
SECT_BG     = (240, 243, 252)

COOL        = ( 33, 120, 220)   # 냉방 / 추워요
WARM        = (220,  72,  35)   # 난방 / 더워요
GREEN       = ( 34, 168,  88)
ORANGE      = (230, 148,  22)
RED         = (200,  50,  52)
TEAL        = ( 18, 185, 152)
PURPLE      = ( 95,  85, 210)

BTN_COLD_BG = ( 33, 120, 220)   # 추워요 버튼
BTN_HOT_BG  = (220,  72,  35)   # 더워요 버튼
BTN_TXT     = (255, 255, 255)

# ── 버튼 영역 (마우스 콜백용) ─────────────────────────────────────────────────
BUTTON_REGIONS: dict = {}   # {'cold': (x1,y1,x2,y2), 'hot': (x1,y1,x2,y2)}


def get_clicked(x: int, y: int):
    """클릭 좌표 → 'cold' | 'hot' | None"""
    for name, (x1, y1, x2, y2) in BUTTON_REGIONS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


# ── 폰트 ──────────────────────────────────────────────────────────────────────
_fc: dict = {}


def _f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    k = (size, bold)
    if k in _fc:
        return _fc[k]
    sys = platform.system()
    if sys == 'Windows':
        root = os.environ.get('SystemRoot', r'C:\Windows')
        cands = [os.path.join(root, 'Fonts', 'malgunbd.ttf' if bold else 'malgun.ttf')]
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
            _fc[k] = f
            return f
        except Exception:
            pass
    f = ImageFont.load_default()
    _fc[k] = f
    return f


# ── 드로잉 헬퍼 ───────────────────────────────────────────────────────────────

def _rrect(draw, x0, y0, x1, y1, r, fill, outline=None, width=1):
    try:
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=r,
                                fill=fill, outline=outline, width=width)
    except AttributeError:
        draw.rectangle([(x0, y0), (x1, y1)], fill=fill, outline=outline, width=width)


def _center_text(draw, cx, y, text, font, fill):
    bb = font.getbbox(text)
    tw = bb[2] - bb[0]
    draw.text((cx - tw // 2, y), text, font=font, fill=fill)


def _badge(draw, x, y, text, bg, fg=(255, 255, 255), r=8):
    fw = _f(13, bold=True)
    bb = fw.getbbox(text)
    bw = bb[2] - bb[0] + 16
    bh = 22
    _rrect(draw, x, y, x + bw, y + bh, r, bg)
    draw.text((x + 8, y + 4), text, font=fw, fill=fg)
    return bw


def _pmv_bar(draw, x, y, pmv):
    """PMV 5단계 세그먼트 바"""
    segs  = [(-99, -1.5, (120, 170, 240), '매우추움'),
             (-1.5, -0.5, (100, 200, 245), '추움'),
             (-0.5,  0.5, ( 50, 190, 110), '쾌적'),
             ( 0.5,  1.5, (250, 165,  50), '더움'),
             ( 1.5,  99,  (220,  70,  70), '매우더움')]
    sw, sh, gap = 62, 8, 3
    cx = x
    active_i = next((i for i, (lo, hi, _, _) in enumerate(segs) if lo <= pmv < hi), 4)
    for i, (_, _, col, lbl) in enumerate(segs):
        dim = tuple(int(c * 0.22 + 235 * 0.78) for c in col)
        fc  = col if i == active_i else dim
        _rrect(draw, cx, y, cx + sw, y + sh, 4, fc)
        if i == active_i:
            lf = _f(11, bold=True)
            lb = lf.getbbox(lbl)
            lw = lb[2] - lb[0]
            draw.text((cx + (sw - lw) // 2, y - 15), lbl, font=lf, fill=col)
        cx += sw + gap


# ── 공개 API ──────────────────────────────────────────────────────────────────

def build(hvac, sm, ds: dict,
          pmv_preference: float, pmv_history: list,
          occ_pred: dict, out_temp: float) -> np.ndarray:
    """
    모바일 앱 스타일 사용자 UI.
    BUTTON_REGIONS 를 갱신하므로 get_clicked() 와 함께 사용.
    """
    _ = pmv_history, occ_pred

    img  = Image.new('RGB', (PANEL_W, PANEL_H), BG)
    draw = ImageDraw.Draw(img)
    PAD  = 16
    CW   = PANEL_W - PAD * 2

    # ────────────────────────────────────────────────────
    # 1. 헤더 (64px)
    # ────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (PANEL_W, 64)], fill=HEADER_BG)
    draw.text((PAD, 8),  'Smart HVAC',   font=_f(20, bold=True), fill=HDR_TXT)
    draw.text((PAD, 36), '스마트 에어컨',  font=_f(14),            fill=HDR_SUB)

    ts = datetime.now().strftime('%H:%M')
    tf = _f(20, bold=True)
    tw = tf.getbbox(ts)[2] - tf.getbbox(ts)[0]
    draw.text((PANEL_W - PAD - tw, 10), ts, font=tf, fill=HDR_TXT)

    smap = {'EMPTY':         ('공실',     TXT_H),
            'ARRIVAL':       ('도착',     (130, 200, 255)),
            'STEADY':        ('운전중',   (100, 240, 150)),
            'LUNCH_BREAK':   ('점심외출', (50,  210, 185)),
            'PRE_DEPARTURE': ('절전',     (255, 190, 80))}
    slbl, scol = smap.get(sm.state.value, (sm.state.value, TXT_H))
    sf = _f(13)
    sw = sf.getbbox(slbl)[2] - sf.getbbox(slbl)[0]
    draw.text((PANEL_W - PAD - sw, 40), slbl, font=sf, fill=scol)

    y = 74

    # ────────────────────────────────────────────────────
    # 2. 온도 카드 (172px)
    # ────────────────────────────────────────────────────
    ch = 172
    _rrect(draw, PAD, y, PAD + CW, y + ch, 16, CARD, BORDER)

    indoor = hvac.indoor_temp
    pmv    = ds.get('pmv_val', 0.0)
    comfort = ds.get('comfort_msg', '-')

    # 온도 초대형
    tc = COOL if indoor < 20 else GREEN if indoor < 27 else WARM
    _center_text(draw, PANEL_W // 2, y + 14, f'{indoor:.1f}°C', _f(52, bold=True), tc)

    # 쾌적도 뱃지
    pc = GREEN if abs(pmv) <= 0.5 else ORANGE if abs(pmv) <= 1.5 else RED
    bw = _badge(draw, 0, 0, comfort[:6], pc)  # measure only
    bw = _f(13, bold=True).getbbox(comfort[:6])[2] - _f(13, bold=True).getbbox(comfort[:6])[0] + 16
    _badge(draw, PANEL_W // 2 - bw // 2, y + 86, comfort[:6], pc)

    # PMV 수치 (작게)
    pv_txt = f'PMV  {pmv:+.2f}'
    pf = _f(13)
    pw = pf.getbbox(pv_txt)[2] - pf.getbbox(pv_txt)[0]
    draw.text((PANEL_W // 2 - pw // 2, y + 112), pv_txt, font=pf, fill=TXT_S)

    # PMV 바
    bar_total = 5 * 62 + 4 * 3
    _pmv_bar(draw, PANEL_W // 2 - bar_total // 2, y + 138, pmv)

    y += ch + 10

    # ────────────────────────────────────────────────────
    # 3. 추워요 / 더워요 버튼 (80px)
    # ────────────────────────────────────────────────────
    bh   = 78
    half = (CW - 10) // 2

    # 추워요 (cold → system heats more)
    bx_c = PAD
    by_c = y
    _rrect(draw, bx_c, by_c, bx_c + half, by_c + bh, 18, BTN_COLD_BG)
    _center_text(draw, bx_c + half // 2, by_c + 10, '추워요',   _f(22, bold=True), BTN_TXT)
    _center_text(draw, bx_c + half // 2, by_c + 46, '따뜻하게', _f(14),            (200, 225, 255))
    BUTTON_REGIONS['cold'] = (bx_c, by_c, bx_c + half, by_c + bh)

    # 더워요 (hot → system cools more)
    bx_h = PAD + half + 10
    _rrect(draw, bx_h, by_c, bx_h + half, by_c + bh, 18, BTN_HOT_BG)
    _center_text(draw, bx_h + half // 2, by_c + 10, '더워요',   _f(22, bold=True), BTN_TXT)
    _center_text(draw, bx_h + half // 2, by_c + 46, '시원하게', _f(14),            (255, 210, 195))
    BUTTON_REGIONS['hot']  = (bx_h, by_c, bx_h + half, by_c + bh)

    y += bh + 8

    # ────────────────────────────────────────────────────
    # 4. 현재 선호 상태 (38px)
    # ────────────────────────────────────────────────────
    ph = 38
    _rrect(draw, PAD, y, PAD + CW, y + ph, 10, SECT_BG, BORDER)
    if pmv_preference > 0:
        plbl, pcol = f'따뜻하게 조절 중  (+{pmv_preference:.1f})', WARM
    elif pmv_preference < 0:
        plbl, pcol = f'시원하게 조절 중  ({pmv_preference:.1f})', COOL
    else:
        plbl, pcol = '선호 설정 없음 — 표준 쾌적도 기준', TXT_S
    pf2 = _f(14)
    pw2 = pf2.getbbox(plbl)[2] - pf2.getbbox(plbl)[0]
    draw.text((PANEL_W // 2 - pw2 // 2, y + 10), plbl, font=pf2, fill=pcol)

    y += ph + 10

    # ────────────────────────────────────────────────────
    # 5. 에어컨 상태 (62px)
    # ────────────────────────────────────────────────────
    ah = 62
    _rrect(draw, PAD, y, PAD + CW, y + ah, 12, CARD, BORDER)

    if not hvac.is_on:
        mode_lbl, mode_col = 'OFF', TXT_H
    elif hvac.mode == 'cool':
        mode_lbl, mode_col = '냉방', COOL
    else:
        mode_lbl, mode_col = '난방', WARM

    draw.text((PAD + 16, y + 10), '에어컨',   font=_f(13), fill=TXT_S)
    draw.text((PAD + 16, y + 30), mode_lbl,  font=_f(18, bold=True), fill=mode_col)

    draw.text((PAD + 120, y + 10), '설정온도', font=_f(13), fill=TXT_S)
    draw.text((PAD + 120, y + 30), f'{hvac.target_temp:.0f}°C', font=_f(18, bold=True), fill=TXT)

    draw.text((PAD + 220, y + 10), '풍량',    font=_f(13), fill=TXT_S)
    # 팬 도트
    for i in range(3):
        fc = (COOL if hvac.mode == 'cool' else WARM) if i < hvac.fan_speed and hvac.is_on else BORDER
        r  = 7
        dx = PAD + 220 + i * 20
        draw.ellipse([(dx, y + 32), (dx + r * 2, y + 32 + r * 2)], fill=fc)

    ppl = ds.get('people_count', 0)
    draw.text((PAD + 330, y + 10), '재실',   font=_f(13), fill=TXT_S)
    draw.text((PAD + 330, y + 30), f'{ppl}명',
              font=_f(18, bold=True), fill=GREEN if ppl > 0 else TXT_H)

    y += ah + 10

    # ────────────────────────────────────────────────────
    # 6. 에너지 절감 (76px)
    # ────────────────────────────────────────────────────
    qh = 76
    _rrect(draw, PAD, y, PAD + CW, y + qh, 12, CARD, BORDER)

    draw.text((PAD + 16, y + 10), '에너지 절감', font=_f(14, bold=True), fill=TXT)

    ai_wh   = ds.get('ai_wh', 0.0)
    saved   = ds.get('rb_wh', 0.0) - ai_wh
    sav_pct = ds.get('savings_pct', 0.0)
    sav_col = GREEN if sav_pct > 0 else (RED if sav_pct < 0 else TXT_S)

    draw.text((PAD + 16,  y + 36), 'AI 소비',  font=_f(12), fill=TXT_S)
    draw.text((PAD + 16,  y + 52), f'{ai_wh:.0f} Wh', font=_f(14), fill=TXT)

    draw.text((PAD + 150, y + 36), '절감량', font=_f(12), fill=TXT_S)
    draw.text((PAD + 150, y + 52), f'{saved:+.0f} Wh', font=_f(14), fill=sav_col)

    draw.text((PAD + 290, y + 36), '절감률', font=_f(12), fill=TXT_S)
    _badge(draw, PAD + 290, y + 52, f'{sav_pct:+.1f}%', sav_col)

    y += qh + 10

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
