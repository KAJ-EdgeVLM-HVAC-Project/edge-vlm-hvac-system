from __future__ import annotations

"""
[시작 화면 모듈]
1단계 : 모드 선택 (실시간 카메라 / 영상 파일 분석)
2단계 : 영상 파일 선택 (영상 모드 선택 시)
3단계 : 환경 프로파일 선택
"""

import cv2
import numpy as np
import platform
import os
import glob
from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
from env_profiles import PROFILES, EnvProfile

W = 960
H = 540

# ── 색상 ──────────────────────────────────────────────────────────────────────
BG        = ( 18,  18,  28)
BG_CARD   = ( 30,  30,  46)
BG_HDR    = ( 26,  22,  52)
BG_SEL    = ( 40,  36,  72)
BORDER    = ( 55,  52,  90)
BORDER_HI = (120, 100, 220)
C_WHITE   = (255, 255, 255)
C_TITLE   = (185, 165, 255)
C_SUB     = (130, 125, 165)
C_HINT    = ( 90,  85, 120)
C_GREEN   = ( 80, 210, 110)
C_ORANGE  = (255, 172,  55)
C_BLUE    = ( 80, 160, 255)

# ── 폰트 캐시 ─────────────────────────────────────────────────────────────────
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
        cands = ['/System/Library/Fonts/AppleSDGothicNeo.ttc']
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
    return ImageFont.load_default()


def _cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _center(draw, cx, y, text, font, fill):
    bb = font.getbbox(text)
    draw.text((cx - (bb[2] - bb[0]) // 2, y), text, font=font, fill=fill)


def _rrect(draw, x0, y0, x1, y1, r, fill, outline=None, width=1):
    try:
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=r,
                                fill=fill, outline=outline, width=width)
    except AttributeError:
        draw.rectangle([(x0, y0), (x1, y1)], fill=fill,
                       outline=outline, width=width)


# ── 반환 타입 ──────────────────────────────────────────────────────────────────
@dataclass
class StartupResult:
    mode: str                    # 'camera' | 'video'
    video_path: Optional[str]    # 영상 파일 경로 (video 모드만)
    profile: EnvProfile          # 환경 프로파일


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1단계: 모드 선택
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MODE_REGIONS: dict = {}

def _render_mode(hover: int) -> np.ndarray:
    img  = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    # 헤더
    draw.rectangle([(0, 0), (W, 90)], fill=BG_HDR)
    _center(draw, W//2, 14, 'Smart HVAC  VLM System', _f(26, True), C_TITLE)
    _center(draw, W//2, 52, 'VLM 기반 지능형 공조 제어 시스템', _f(14), C_SUB)

    # 모드 카드 2개
    cards = [
        (0, '실시간 카메라',    '카메라 영상을 실시간 분석\n즉각적인 HVAC 제어', C_BLUE),
        (1, '영상 파일 분석',   '저장된 영상으로 분석\n논문용 데이터 · 그래프 자동 생성', C_GREEN),
    ]
    _MODE_REGIONS.clear()
    CW, CH = 390, 260
    gap = (W - CW * 2) // 3
    cy0 = 130

    for idx, title, desc, col in cards:
        x0 = gap + idx * (CW + gap)
        x1 = x0 + CW
        y1 = cy0 + CH

        is_hi = (hover == idx)
        bg    = BG_SEL if is_hi else BG_CARD
        bord  = BORDER_HI if is_hi else BORDER
        _rrect(draw, x0, cy0, x1, y1, 16, bg, bord, width=2 if is_hi else 1)

        # 상단 색상 바
        _rrect(draw, x0, cy0, x1, cy0 + 6, 4, col)

        # 숫자 뱃지
        num = str(idx + 1)
        _rrect(draw, x0 + 16, cy0 + 22, x0 + 46, cy0 + 52, 14,
               col if is_hi else BORDER)
        _center(draw, x0 + 31, cy0 + 24, num, _f(18, True), C_WHITE)

        # 제목
        draw.text((x0 + 60, cy0 + 24), title, font=_f(20, True),
                  fill=C_WHITE if is_hi else (180, 175, 215))

        # 설명
        dy = cy0 + 70
        for line in desc.split('\n'):
            _center(draw, x0 + CW//2, dy, line, _f(13), C_SUB)
            dy += 22

        # 버튼
        btn_col = col if is_hi else BORDER
        _rrect(draw, x0 + 20, y1 - 50, x1 - 20, y1 - 16, 10, btn_col)
        _center(draw, x0 + CW//2, y1 - 46,
                '선택  →' if is_hi else '클릭하여 선택', _f(14, True),
                C_WHITE if is_hi else C_HINT)

        _MODE_REGIONS[idx] = (x0, cy0, x1, y1)

    # 힌트
    _center(draw, W//2, H - 30, '키보드: 1 = 카메라    2 = 영상 파일    ESC = 기본값(카메라)',
            _f(12), C_HINT)
    return _cv(img)


def _select_mode() -> str:
    """모드 선택 화면 — 'camera' 또는 'video' 반환"""
    selected = [None]
    hover    = [-1]

    def on_mouse(event, x, y, flags, _):
        h = -1
        for idx, (x0, y0, x1, y1) in _MODE_REGIONS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                h = idx
                break
        hover[0] = h
        if event == cv2.EVENT_LBUTTONDOWN and h >= 0:
            selected[0] = ['camera', 'video'][h]

    WIN = 'Smart HVAC - 모드 선택'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    for _ in range(5):
        cv2.waitKey(1)
    cv2.setMouseCallback(WIN, on_mouse)

    while selected[0] is None:
        cv2.imshow(WIN, _render_mode(hover[0]))
        k = cv2.waitKey(30) & 0xFF
        if k == ord('1'):
            selected[0] = 'camera'
        elif k == ord('2'):
            selected[0] = 'video'
        elif k == 27:
            selected[0] = 'camera'
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            selected[0] = 'camera'

    cv2.destroyWindow(WIN)
    return selected[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2단계: 영상 파일 선택
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm')

def _find_videos() -> list[str]:
    """현재 디렉터리와 하위 2단계에서 영상 파일 목록 반환"""
    results = []
    for ext in VIDEO_EXTS:
        results += glob.glob(f'**/*{ext}', recursive=False)
        results += glob.glob(f'*{ext}')
        results += glob.glob(f'videos/*{ext}')
        results += glob.glob(f'data/*{ext}')
    seen = set()
    out  = []
    for p in results:
        p = os.path.normpath(p)
        if p not in seen:
            seen.add(p)
            if os.path.isfile(p):
                out.append(p)
    return sorted(out)


def _render_filepicker(files: list[str], cursor: int, typed: str) -> np.ndarray:
    img  = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)

    # 헤더
    draw.rectangle([(0, 0), (W, 72)], fill=BG_HDR)
    _center(draw, W//2, 10, '영상 파일 선택', _f(22, True), C_TITLE)
    _center(draw, W//2, 44, '분석할 영상 파일을 선택하세요', _f(13), C_SUB)

    # 파일 목록 영역
    LIST_Y = 84
    ROW_H  = 38
    MAX_ROWS = 9
    draw.rectangle([(16, LIST_Y), (W - 16, LIST_Y + ROW_H * MAX_ROWS + 8)],
                   fill=BG_CARD, outline=BORDER)

    start = max(0, cursor - MAX_ROWS + 1)
    display = files[start: start + MAX_ROWS]

    for i, path in enumerate(display):
        gy      = LIST_Y + 4 + i * ROW_H
        real_i  = start + i
        is_sel  = (real_i == cursor)

        if is_sel:
            draw.rectangle([(17, gy), (W - 17, gy + ROW_H - 2)], fill=BG_SEL)
            draw.rectangle([(17, gy), (19, gy + ROW_H - 2)], fill=C_BLUE)

        # 파일명 + 크기
        name = os.path.basename(path)
        size = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        draw.text((28, gy + 8), name, font=_f(14, is_sel),
                  fill=C_WHITE if is_sel else (175, 170, 210))
        draw.text((W - 110, gy + 10), f'{size:.1f} MB',
                  font=_f(12), fill=C_SUB)
        # 상위 경로
        parent = os.path.dirname(path)
        if parent:
            draw.text((28 + _f(14).getbbox(name)[2] + 8, gy + 10),
                      f'  ({parent})', font=_f(12), fill=C_HINT)

    if not files:
        _center(draw, W//2, LIST_Y + ROW_H * 3,
                '현재 폴더에 영상 파일이 없습니다', _f(15), C_SUB)

    # 직접 입력 칸
    input_y = LIST_Y + ROW_H * MAX_ROWS + 20
    draw.text((16, input_y), '직접 입력:', font=_f(14), fill=C_SUB)
    _rrect(draw, 100, input_y - 4, W - 16, input_y + 26, 6,
           BG_CARD, BORDER_HI, width=1)
    display_text = typed[-60:] if len(typed) > 60 else typed
    draw.text((108, input_y), display_text + '|', font=_f(14), fill=C_WHITE)

    # 힌트
    hint_y = H - 44
    draw.rectangle([(0, hint_y - 6), (W, H)], fill=(22, 18, 38))
    _center(draw, W//2, hint_y,
            '↑↓: 선택   Enter: 확인   ESC: 뒤로   파일 경로 직접 타이핑 가능',
            _f(12), C_HINT)

    return _cv(img)


def _select_video() -> Optional[str]:
    """영상 파일 선택 화면 — 경로 문자열 또는 None(취소) 반환"""
    files  = _find_videos()
    cursor = [0]
    typed  = ['']

    WIN = 'Smart HVAC - 영상 파일 선택'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    for _ in range(5):
        cv2.waitKey(1)

    while True:
        cv2.imshow(WIN, _render_filepicker(files, cursor[0], typed[0]))
        k = cv2.waitKey(50) & 0xFF

        if k == 27:                         # ESC → 취소
            cv2.destroyWindow(WIN)
            return None

        elif k == 13 or k == 10:            # Enter → 확인
            # 직접 입력 우선
            if typed[0].strip():
                p = typed[0].strip()
                if os.path.isfile(p):
                    cv2.destroyWindow(WIN)
                    return p
                else:
                    typed[0] = ''           # 잘못된 경로 → 초기화
            elif files:
                cv2.destroyWindow(WIN)
                return files[cursor[0]]

        elif k == 82 or k == 0:             # 위쪽 화살표
            cursor[0] = max(0, cursor[0] - 1)

        elif k == 84 or k == 1:             # 아래쪽 화살표
            cursor[0] = min(len(files) - 1, cursor[0] + 1)

        elif k == 8 or k == 127:            # Backspace
            typed[0] = typed[0][:-1]

        elif 32 <= k <= 126:                # 출력 가능 ASCII
            typed[0] += chr(k)

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(WIN)
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3단계: 환경 프로파일 선택 (기존 로직 유지 + 키보드 단축키 추가)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CARD_REGIONS: dict = {}


def _icon_office(draw, cx, cy, col):
    draw.rectangle([(cx-26, cy-28), (cx+26, cy+24)], fill=col)
    for wx in [cx-18, cx-4, cx+10]:
        for wy in [cy-22, cy-8]:
            draw.rectangle([(wx, wy), (wx+8, wy+9)], fill=(255,255,255))
    draw.rectangle([(cx-7, cy+8), (cx+7, cy+24)], fill=(255,255,255))


def _icon_home(draw, cx, cy, col):
    draw.polygon([(cx, cy-32), (cx-28, cy-8), (cx+28, cy-8)], fill=col)
    draw.rectangle([(cx-22, cy-8), (cx+22, cy+24)], fill=col)
    draw.rectangle([(cx-8,  cy+8), (cx+8,  cy+24)], fill=(255,255,255))
    draw.rectangle([(cx-18, cy-4), (cx-8,  cy+6)],  fill=(255,255,255))
    draw.rectangle([(cx+8,  cy-4), (cx+18, cy+6)],  fill=(255,255,255))


def _icon_gym(draw, cx, cy, col):
    draw.rectangle([(cx-26, cy-4), (cx+26, cy+4)], fill=col)
    draw.rectangle([(cx-38, cy-15), (cx-26, cy+15)], fill=col)
    draw.rectangle([(cx-46, cy-10), (cx-36, cy+10)], fill=col)
    draw.rectangle([(cx+26, cy-15), (cx+38, cy+15)], fill=col)
    draw.rectangle([(cx+36, cy-10), (cx+46, cy+10)], fill=col)


def _icon_facility(draw, cx, cy, col):
    pts = [(cx, cy-32), (cx+24, cy-18), (cx+24, cy+4),
           (cx, cy+26), (cx-24, cy+4), (cx-24, cy-18)]
    draw.polygon(pts, fill=col)
    draw.line([(cx-10, cy+2), (cx-2, cy+10), (cx+12, cy-8)],
              fill=(255,255,255), width=4)


_ICON_FN = {
    'office':   _icon_office,
    'home':     _icon_home,
    'gym':      _icon_gym,
    'facility': _icon_facility,
}


def _render_profile(hover_key: str | None) -> np.ndarray:
    img  = Image.new('RGB', (W, H), (245, 247, 252))
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (W, 80)], fill=(17, 24, 52))
    draw.text((32, 12), 'Smart HVAC',
              font=_f(24, True), fill=(255,255,255))
    draw.text((32, 46), '사용 공간을 선택하면 최적화된 제어 파라미터가 자동 적용됩니다.',
              font=_f(14), fill=(140, 150, 200))

    keys = list(PROFILES.keys())
    n    = len(keys)
    PAD  = 18
    CW   = (W - PAD * (n + 1)) // n
    CH   = 370
    cy0  = 96

    CARD_REGIONS.clear()
    for i, key in enumerate(keys):
        prof = PROFILES[key]
        cx0  = PAD + i * (CW + PAD)
        cx1  = cx0 + CW
        cy1  = cy0 + CH
        CARD_REGIONS[key] = (cx0, cy0, cx1, cy1)

        is_hi  = (key == hover_key)
        shadow = (200, 205, 220) if is_hi else (220, 223, 235)
        _rrect(draw, cx0, cy0, cx1, cy1, 14,
               (252,253,255) if is_hi else (255,255,255),
               shadow, width=2 if is_hi else 1)
        col = prof.icon_color
        _rrect(draw, cx0, cy0, cx1, cy0+6, 4, col)

        # 숫자 뱃지
        _rrect(draw, cx0+10, cy0+12, cx0+34, cy0+36, 10,
               col if is_hi else (220,222,235))
        _center(draw, cx0+22, cy0+13, str(i+1), _f(14, True),
                (255,255,255) if is_hi else (150,150,170))

        icon_cx = cx0 + CW // 2
        _ICON_FN[key](draw, icon_cx, cy0 + 60, col)
        _center(draw, icon_cx, cy0+112, prof.name, _f(22, True), (22,22,36))
        _center(draw, icon_cx, cy0+144, prof.desc, _f(13), (98,102,125))
        draw.line([(cx0+16, cy0+166), (cx1-16, cy0+166)],
                  fill=(220,223,235), width=1)
        fy = cy0 + 178
        for feat in prof.features:
            draw.text((cx0+18, fy), f'·  {feat}', font=_f(13), fill=(98,102,125))
            fy += 22
        btn_col = col if is_hi else (230, 232, 240)
        btn_txt = (255,255,255) if is_hi else (165,168,185)
        _rrect(draw, cx0+14, cy1-44, cx1-14, cy1-14, 10, btn_col)
        _center(draw, icon_cx, cy1-40, '탭하여 선택', _f(14, True), btn_txt)

    hint = f'키보드 숫자 1~{n} 로 빠르게 선택 가능  |  ESC = 기본값(사무실)'
    hb   = _f(12).getbbox(hint)
    draw.text((W//2 - (hb[2]-hb[0])//2, H-26), hint, font=_f(12),
              fill=(165,168,185))
    return _cv(img)


def _select_profile() -> EnvProfile:
    selected  = [None]
    hover_key = [None]

    def on_mouse(event, x, y, flags, _):
        hk = None
        for key, (x1, y1, x2, y2) in CARD_REGIONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                hk = key
                break
        hover_key[0] = hk
        if event == cv2.EVENT_LBUTTONDOWN and hk:
            selected[0] = hk

    WIN = 'Smart HVAC - 환경 설정'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)
    for _ in range(5):
        cv2.waitKey(1)
    cv2.setMouseCallback(WIN, on_mouse)

    keys = list(PROFILES.keys())
    while selected[0] is None:
        cv2.imshow(WIN, _render_profile(hover_key[0]))
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            selected[0] = 'office'
        # 숫자 키 1~N
        elif ord('1') <= k <= ord('9'):
            idx = k - ord('1')
            if idx < len(keys):
                selected[0] = keys[idx]
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            selected[0] = 'office'

    cv2.destroyWindow(WIN)
    prof = PROFILES[selected[0]]
    print(f'[환경] 선택됨: {prof.name} ({prof.key})')
    return prof


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공개 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WIN_NAME = 'Smart HVAC - 환경 설정'   # 하위 호환용


def show_and_select() -> StartupResult:
    """
    3단계 시작 화면을 순서대로 표시하고 StartupResult 반환.
    ESC / 창 닫기 시 기본값(camera + office) 반환.
    """
    # 1단계: 모드
    mode = _select_mode()

    # 2단계: 영상 파일 (video 모드일 때만)
    video_path = None
    if mode == 'video':
        video_path = _select_video()
        if video_path is None:
            # 취소 → 카메라 모드로 전환
            mode = 'camera'

    # 3단계: 환경 프로파일
    profile = _select_profile()

    return StartupResult(mode=mode, video_path=video_path, profile=profile)
