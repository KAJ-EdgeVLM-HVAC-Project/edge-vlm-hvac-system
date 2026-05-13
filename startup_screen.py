from __future__ import annotations

"""
[환경 선택 화면]
시스템 시작 시 사용 공간 유형을 선택합니다.
마우스 호버 + 클릭으로 선택.
"""

import cv2
import numpy as np
import platform
import os
from PIL import Image, ImageDraw, ImageFont
from env_profiles import PROFILES, EnvProfile

W = 960
H = 520

CARD_REGIONS: dict = {}   # {key: (x1,y1,x2,y2)}

# ── 색상 ──────────────────────────────────────────────────────────────────────
BG         = (245, 247, 252)
HEADER_BG  = (17,  24,  52)
CARD_BG    = (255, 255, 255)
BORDER     = (220, 223, 235)
TXT        = (22,  22,  36)
TXT_S      = (98, 102, 125)
TXT_H      = (165, 168, 185)

# ── 폰트 ──────────────────────────────────────────────────────────────────────
_fc: dict = {}


def _f(size: int, bold: bool = False):
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


def _rrect(draw, x0, y0, x1, y1, r, fill, outline=None, width=1):
    try:
        draw.rounded_rectangle([(x0, y0), (x1, y1)],
                                radius=r, fill=fill, outline=outline, width=width)
    except AttributeError:
        draw.rectangle([(x0, y0), (x1, y1)], fill=fill, outline=outline, width=width)


def _center(draw, cx, y, text, font, fill):
    bb = font.getbbox(text)
    draw.text((cx - (bb[2] - bb[0]) // 2, y), text, font=font, fill=fill)


# ── 환경별 아이콘 ──────────────────────────────────────────────────────────────

def _icon_office(draw, cx, cy, col):
    """빌딩 아이콘"""
    # 건물 본체
    draw.rectangle([(cx-26, cy-28), (cx+26, cy+24)], fill=col)
    # 창문 6개
    for wx in [cx-18, cx-4, cx+10]:
        for wy in [cy-22, cy-8]:
            draw.rectangle([(wx, wy), (wx+8, wy+9)], fill=(255, 255, 255))
    # 출입문
    draw.rectangle([(cx-7, cy+8), (cx+7, cy+24)], fill=(255, 255, 255))


def _icon_home(draw, cx, cy, col):
    """집 아이콘"""
    # 지붕 (삼각형)
    draw.polygon([(cx, cy-32), (cx-28, cy-8), (cx+28, cy-8)], fill=col)
    # 벽
    draw.rectangle([(cx-22, cy-8), (cx+22, cy+24)], fill=col)
    # 문
    draw.rectangle([(cx-8, cy+8), (cx+8, cy+24)], fill=(255, 255, 255))
    # 창문
    draw.rectangle([(cx-18, cy-4), (cx-8,  cy+6)],  fill=(255, 255, 255))
    draw.rectangle([(cx+8,  cy-4), (cx+18, cy+6)],  fill=(255, 255, 255))


def _icon_gym(draw, cx, cy, col):
    """덤벨 아이콘"""
    # 바
    draw.rectangle([(cx-26, cy-4), (cx+26, cy+4)], fill=col)
    # 왼쪽 웨이트 (두 겹)
    draw.rectangle([(cx-38, cy-15), (cx-26, cy+15)], fill=col)
    draw.rectangle([(cx-46, cy-10), (cx-36, cy+10)], fill=col)
    # 오른쪽 웨이트 (두 겹)
    draw.rectangle([(cx+26, cy-15), (cx+38, cy+15)], fill=col)
    draw.rectangle([(cx+36, cy-10), (cx+46, cy+10)], fill=col)


def _icon_facility(draw, cx, cy, col):
    """방패 아이콘"""
    pts = [(cx, cy-32), (cx+24, cy-18),
           (cx+24, cy+4), (cx, cy+26),
           (cx-24, cy+4), (cx-24, cy-18)]
    draw.polygon(pts, fill=col)
    # 별 대신 간단한 마크 (V 체크)
    draw.line([(cx-10, cy+2), (cx-2, cy+10), (cx+12, cy-8)],
              fill=(255, 255, 255), width=4)


_ICON_FN = {
    'office':   _icon_office,
    'home':     _icon_home,
    'gym':      _icon_gym,
    'facility': _icon_facility,
}


# ── 화면 렌더링 ───────────────────────────────────────────────────────────────

def _render(hover_key: str | None = None) -> np.ndarray:
    img  = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (W, 80)], fill=HEADER_BG)
    draw.text((32, 12), 'Smart HVAC',
              font=_f(24, bold=True), fill=(255, 255, 255))
    draw.text((32, 46), '사용 공간을 선택하면 최적화된 제어 파라미터가 자동 적용됩니다.',
              font=_f(14), fill=(140, 150, 200))

    # ── 카드 ─────────────────────────────────────────────────────────────────
    keys   = list(PROFILES.keys())
    n      = len(keys)
    PAD    = 18
    CW     = (W - PAD * (n + 1)) // n   # 카드 너비
    CH     = 370                         # 카드 높이
    card_y = 96

    CARD_REGIONS.clear()

    for i, key in enumerate(keys):
        prof = PROFILES[key]
        cx0  = PAD + i * (CW + PAD)
        cy0  = card_y
        cx1  = cx0 + CW
        cy1  = cy0 + CH

        CARD_REGIONS[key] = (cx0, cy0, cx1, cy1)

        is_hover = (key == hover_key)
        shadow   = (200, 205, 220) if is_hover else BORDER
        card_col = (252, 253, 255) if is_hover else CARD_BG

        # 카드 배경 (호버 시 살짝 강조)
        _rrect(draw, cx0, cy0, cx1, cy1, 14, card_col, shadow,
               width=2 if is_hover else 1)

        # 상단 포인트 바
        col = prof.icon_color
        _rrect(draw, cx0, cy0, cx1, cy0 + 6, 4, col)

        # 아이콘
        icon_cx = cx0 + CW // 2
        icon_cy = cy0 + 60
        _ICON_FN[key](draw, icon_cx, icon_cy, col)

        # 이름
        _center(draw, icon_cx, cy0 + 112,
                prof.name, _f(22, bold=True), TXT)

        # 설명
        _center(draw, icon_cx, cy0 + 144,
                prof.desc, _f(13), TXT_S)

        # 구분선
        draw.line([(cx0 + 16, cy0 + 166), (cx1 - 16, cy0 + 166)],
                  fill=BORDER, width=1)

        # 특징 목록
        fy = cy0 + 178
        for feat in prof.features:
            draw.text((cx0 + 18, fy), f'·  {feat}',
                      font=_f(13), fill=TXT_S)
            fy += 22

        # 하단 선택 버튼
        btn_col = col if is_hover else (230, 232, 240)
        btn_txt = (255, 255, 255) if is_hover else TXT_H
        _rrect(draw, cx0 + 14, cy1 - 44, cx1 - 14, cy1 - 14,
               10, btn_col)
        _center(draw, icon_cx, cy1 - 40,
                '탭하여 선택', _f(14, bold=True), btn_txt)

    # ── 하단 힌트 ─────────────────────────────────────────────────────────────
    hint = 'VLM 분석 결과가 있으면 항상 VLM 값이 우선 적용됩니다.  프로파일은 VLM 실패 시 fallback 으로만 사용됩니다.'
    hf  = _f(12)
    hb  = hf.getbbox(hint)
    draw.text((W // 2 - (hb[2] - hb[0]) // 2, H - 26),
              hint, font=hf, fill=TXT_H)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── 공개 API ──────────────────────────────────────────────────────────────────

WIN_NAME = 'Smart HVAC - 환경 설정'


def show_and_select() -> EnvProfile:
    """
    환경 선택 화면을 표시하고 클릭된 프로파일을 반환합니다.
    ESC 또는 창 닫기 → 사무실(office) 기본값 반환.
    """
    selected  = [None]
    hover_key = [None]

    def on_mouse(event, x, y, flags, _param):
        # 호버 업데이트
        hk = None
        for key, (x1, y1, x2, y2) in CARD_REGIONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                hk = key
                break
        hover_key[0] = hk

        if event == cv2.EVENT_LBUTTONDOWN and hk:
            selected[0] = hk

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.waitKey(1)  # Qt 이벤트 루프 초기화 대기 (없으면 setMouseCallback NULL handler)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    while selected[0] is None:
        frame = _render(hover_key[0])
        cv2.imshow(WIN_NAME, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:                   # ESC → 기본값
            selected[0] = 'office'
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            selected[0] = 'office'     # 창 닫기 → 기본값

    cv2.destroyWindow(WIN_NAME)
    prof = PROFILES[selected[0]]
    print(f'[환경] 선택됨: {prof.name} ({prof.key})')
    return prof
