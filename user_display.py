"""
[사용자 인터페이스 패널]
운영자 대시보드와 별도 창으로 표시되는 사용자 중심 UI.

── 구성 섹션 ──────────────────────────────────────────
1. 현재 실내 온도 + PMV + 사용자 선호 조절 (U: 따뜻 / D: 시원)
2. 공기질 5단계 (PM10 / PM2.5 / 통합지수)
3. 창문 솔루션 메시지 (상황 인식형)
4. PMV 보정 이력 그래프 (선호 기준선 포함)
5. 에너지 절약 실시간 카운터 (kWh / CO₂ / 절약금액)
6. 재실 예측 알림 (CSV 패턴 기반)
"""

import cv2
import numpy as np
import platform
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 700
PANEL_H = 940

# ── 색상 팔레트 (RGB) ─────────────────────────────────────────────────────────
BG          = ( 14,  18,  34)
BG_SECT     = ( 22,  28,  48)
BG_HDR      = ( 35,  42,  72)
C_TITLE     = (180, 200, 255)
C_LABEL     = (100, 110, 145)
C_VAL       = (210, 220, 240)
C_GREEN     = ( 80, 200,  90)
C_TEAL      = ( 70, 200, 180)
C_ORANGE    = (255, 165,  50)
C_RED       = (210,  70,  70)
C_GOLD      = (255, 215,  70)
C_BLUE      = ( 80, 160, 255)
C_PURPLE    = (160, 130, 255)
C_TIME      = (110, 115, 155)
C_HEAT      = (255, 148,  55)
C_COOL      = ( 72, 165, 255)
C_VGOOD     = ( 50, 220, 160)   # 아주좋음

# ── 폰트 캐시 ─────────────────────────────────────────────────────────────────
_font_cache: dict = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    sys_name = platform.system()
    if sys_name == 'Windows':
        root = os.environ.get('SystemRoot', r'C:\Windows')
        cands = [os.path.join(root, 'Fonts', 'malgunbd.ttf' if bold else 'malgun.ttf'),
                 os.path.join(root, 'Fonts', 'malgun.ttf')]
    elif sys_name == 'Darwin':
        cands = ['/System/Library/Fonts/AppleSDGothicNeo.ttc',
                 '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
                 '/Library/Fonts/NanumGothic.ttf']
    else:
        cands = ['/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf' if bold
                 else '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
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


# ── 공기질 5단계 ──────────────────────────────────────────────────────────────

def _aq_pm10(pm10: int) -> tuple:
    if pm10 <= 15:  return '아주좋음', C_VGOOD
    if pm10 <= 30:  return '좋음',     C_GREEN
    if pm10 <= 80:  return '보통',     C_VAL
    if pm10 <= 150: return '나쁨',     C_ORANGE
    return '매우나쁨', C_RED


def _aq_pm25(pm25: int) -> tuple:
    if pm25 <= 8:   return '아주좋음', C_VGOOD
    if pm25 <= 15:  return '좋음',     C_GREEN
    if pm25 <= 35:  return '보통',     C_VAL
    if pm25 <= 75:  return '나쁨',     C_ORANGE
    return '매우나쁨', C_RED


def _aq_khai(khai) -> tuple:
    try:
        v = int(khai)
    except (TypeError, ValueError):
        return '-', C_LABEL
    if v <= 1:  return '좋음',     C_GREEN
    if v <= 2:  return '보통',     C_VAL
    if v <= 3:  return '나쁨',     C_ORANGE
    return '매우나쁨', C_RED


# ── 창문 솔루션 메시지 ────────────────────────────────────────────────────────

def _window_advice(hvac, out_temp: float, pm10: int,
                   pmv: float, people: int) -> tuple:
    """(메시지, 색상) — 상황 인식형"""
    if people == 0:
        return "공실 — 창문 상태 유지", C_LABEL

    if pm10 > 150:
        return f"미세먼지 매우나쁨({pm10}㎍) — 창문 즉시 닫으세요", C_RED
    if pm10 > 80:
        return f"미세먼지 나쁨({pm10}㎍) — 창문 닫으세요", C_ORANGE

    if hvac.is_on:
        mode_kor = "난방" if hvac.mode == 'heat' else "냉방"
        return f"{mode_kor} 가동 중 — 에너지 손실 방지, 창문 닫으세요", C_ORANGE

    indoor = hvac.indoor_temp
    if pmv > 1.0 and out_temp < indoor - 2.0:
        return f"덥고 실외 시원({out_temp:.1f}°C) — 창문 열어 자연 환기", C_TEAL
    if pmv > 0.5 and out_temp < indoor - 3.0:
        return f"실외 {out_temp:.1f}°C — 창문 열어 보조 냉방 권장", C_TEAL
    if abs(pmv) <= 0.5 and pm10 <= 30:
        return f"공기질 좋음, 쾌적 온도 — 창문 환기 권장", C_GREEN

    return f"실내 {indoor:.1f}°C — 창문 현재 상태 유지", C_VAL


# ── PMV 이력 그래프 ───────────────────────────────────────────────────────────

def _draw_pmv_graph(draw: ImageDraw.Draw, x0: int, y0: int,
                    gw: int, gh: int,
                    pmv_history: list, pmv_preference: float) -> None:
    PMV_MIN, PMV_MAX = -3.0, 3.0
    rng = PMV_MAX - PMV_MIN

    # 그래프 배경 + 테두리
    draw.rectangle([(x0, y0), (x0 + gw, y0 + gh)], fill=(18, 22, 40))
    draw.rectangle([(x0, y0), (x0 + gw, y0 + gh)], outline=(55, 62, 100), width=1)

    def _py(pmv_v):
        return int(y0 + (PMV_MAX - max(PMV_MIN, min(PMV_MAX, pmv_v))) / rng * gh)

    # 쾌적 구간 배경 (-0.5 ~ +0.5)
    draw.rectangle([(x0, _py(0.5)), (x0 + gw, _py(-0.5))], fill=(25, 65, 40))

    # 격자선
    for v in [-2, -1, 0, 1, 2]:
        gy = _py(v)
        lw = 2 if v == 0 else 1
        col = (55, 90, 60) if v == 0 else (45, 50, 80)
        draw.line([(x0, gy), (x0 + gw, gy)], fill=col, width=lw)
        draw.text((x0 + 3, gy - 9), f'{v:+d}', font=_font(11), fill=(80, 90, 130))

    # 사용자 선호 기준선
    pref_y = _py(pmv_preference)
    draw.line([(x0, pref_y), (x0 + gw, pref_y)], fill=(200, 160, 60), width=2)
    draw.text((x0 + gw - 72, pref_y - 14),
              f'선호:{pmv_preference:+.1f}', font=_font(12), fill=(200, 160, 60))

    # 데이터 없으면 안내문
    if not pmv_history:
        draw.text((x0 + gw // 2 - 50, y0 + gh // 2 - 8),
                  '데이터 수집 중...', font=_font(13), fill=C_LABEL)
        return

    # PMV 선 그리기
    n    = len(pmv_history)
    step = gw / max(n, 1)
    pts  = [(int(x0 + i * step + step / 2), _py(v))
            for i, v in enumerate(pmv_history)]

    for i in range(len(pts) - 1):
        v   = pmv_history[i]
        col = (C_GREEN if abs(v) <= 0.5 else
               C_ORANGE if abs(v) <= 1.5 else C_RED)
        draw.line([pts[i], pts[i + 1]], fill=col, width=2)

    # 마지막 점 강조
    lx, ly = pts[-1]
    draw.ellipse([(lx - 4, ly - 4), (lx + 4, ly + 4)], fill=C_GOLD)
    draw.text((x0 + gw - 75, y0 + 3),
              f'현재:{pmv_history[-1]:+.2f}', font=_font(13), fill=C_GOLD)


# ── 에너지 절약 섹션 ──────────────────────────────────────────────────────────

def _draw_energy(draw: ImageDraw.Draw, y: int, em_data: dict) -> int:
    h_total = 168
    draw.rectangle([(0, y), (PANEL_W, y + h_total)], fill=BG_SECT)
    draw.rectangle([(0, y), (PANEL_W, y + 38)], fill=(30, 48, 32))
    draw.text((14, y + 8), '  에너지 절약 카운터',
              font=_font(19, bold=True), fill=C_GREEN)
    y += 38

    saved_kwh   = em_data.get('saved_kwh',   0.0)
    actual_kwh  = em_data.get('actual_kwh',  0.0)
    savings_pct = em_data.get('savings_pct', 0.0)
    comfort_pct = em_data.get('comfort_pct', 0.0)
    co2_kg      = round(max(0.0, saved_kwh) * 0.4545, 4)   # 한국 전력 탄소 계수
    money_won   = int(max(0.0, saved_kwh) * 120)            # 절약 금액 (원, 상업용 요금 기준)

    s_col = C_GREEN if saved_kwh >= 0 else C_RED

    # 행 1: 절약량 (크게)
    draw.text(( 14, y + 4), '절약량',    font=_font(16),           fill=C_LABEL)
    draw.text(( 90, y + 1), f'{saved_kwh:.4f} kWh',
              font=_font(22, bold=True), fill=s_col)
    draw.text((380, y + 4), '현재소비',  font=_font(16),           fill=C_LABEL)
    draw.text((460, y + 4), f'{actual_kwh:.4f} kWh',
              font=_font(17),           fill=C_VAL)
    y += 38

    # 행 2: CO₂ + 금액
    draw.text(( 14, y + 4), 'CO₂ 감축', font=_font(16),           fill=C_LABEL)
    draw.text((100, y + 4), f'{co2_kg:.3f} kg',
              font=_font(19, bold=True), fill=C_TEAL)
    draw.text((350, y + 4), '절약금액', font=_font(16),            fill=C_LABEL)
    draw.text((430, y + 4), f'{money_won:,}원',
              font=_font(19, bold=True), fill=C_GOLD)
    y += 38

    # 행 3: 절약률 + 쾌적유지율
    draw.text(( 14, y + 4), '절약률', font=_font(16), fill=C_LABEL)
    draw.text(( 80, y + 4), f'{savings_pct:.1f}%',
              font=_font(19, bold=True),
              fill=C_GREEN if savings_pct >= 0 else C_RED)
    draw.text((350, y + 4), '쾌적유지', font=_font(16), fill=C_LABEL)
    draw.text((430, y + 4), f'{comfort_pct:.1f}%',
              font=_font(19, bold=True),
              fill=C_GREEN if comfort_pct >= 70 else C_ORANGE)
    y += 38

    # 베이스라인 설명 (작게)
    draw.text((14, y + 4),
              '※ 베이스라인: 재실 시 Fan2(1200W) 상시 가동 기준',
              font=_font(13), fill=C_LABEL)
    y += 16
    return y


# ── 재실 예측 섹션 ────────────────────────────────────────────────────────────

def _draw_occ_pred(draw: ImageDraw.Draw, y: int, occ_pred: dict) -> int:
    h_total = 88
    draw.rectangle([(0, y), (PANEL_W, y + h_total)], fill=BG_SECT)
    draw.rectangle([(0, y), (PANEL_W, y + 38)], fill=(38, 32, 62))
    draw.text((14, y + 8), '  재실 예측 (CSV 패턴 분석)',
              font=_font(19, bold=True), fill=C_PURPLE)
    y += 38

    msg   = occ_pred.get('message', '데이터 수집 중...')
    col   = occ_pred.get('color',   C_LABEL)
    count = occ_pred.get('record_count', 0)

    draw.text((14, y + 4),  f'▶ {msg}', font=_font(16), fill=col)
    y += 28
    draw.text((14, y + 4),  f'   (로그 {count}건 기반)', font=_font(13), fill=C_LABEL)
    y += 22
    return y


# ── 공개 API ──────────────────────────────────────────────────────────────────

def build(hvac, sm, ds: dict, em_data: dict,
          pmv_preference: float, pmv_history: list,
          occ_pred: dict, out_temp: float) -> np.ndarray:
    """
    사용자 인터페이스 패널 생성.

    Args:
        hvac           : HVACSimulator 인스턴스
        sm             : StateManager 인스턴스
        ds             : display_state dict
        em_data        : {'saved_kwh', 'actual_kwh', 'savings_pct', 'comfort_pct'}
        pmv_preference : 사용자 PMV 선호 오프셋  (-2.0 ~ +2.0)
        pmv_history    : 최근 PMV 값 리스트 (최대 30개)
        occ_pred       : {'message', 'color', 'record_count'}
        out_temp       : 실외 기온 (°C)

    Returns:
        np.ndarray (BGR, PANEL_W × PANEL_H)
    """
    img  = Image.new('RGB', (PANEL_W, PANEL_H), BG)
    draw = ImageDraw.Draw(img)

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (PANEL_W, 58)], fill=(22, 28, 58))
    draw.text((14,  5),  '사용자 인터페이스',
              font=_font(24, bold=True), fill=C_TITLE)
    state_kor = {'EMPTY': '공실', 'ARRIVAL': '도착',
                 'STEADY': '안정', 'PRE_DEPARTURE': '퇴실준비'}
    state_txt = state_kor.get(sm.state.value, sm.state.value)
    draw.text((14, 38), datetime.now().strftime('%H:%M:%S') + f'  [{state_txt}]',
              font=_font(15), fill=C_TIME)
    y = 58

    # ── 1. 현재 실내 환경 + 사용자 선호 ─────────────────────────────────────
    indoor_temp = hvac.indoor_temp
    pmv_val     = ds.get('pmv_val', 0.0)
    comfort_msg = ds.get('comfort_msg', '-')

    draw.rectangle([(0, y), (PANEL_W, y + 148)], fill=BG_SECT)
    draw.rectangle([(0, y), (PANEL_W, y + 38)],  fill=BG_HDR)
    draw.text((14, y + 8), '  현재 실내 환경',
              font=_font(19, bold=True), fill=C_TITLE)
    y += 38

    # 온도 대형
    t_col = (C_COOL if indoor_temp < 19 else
             C_GREEN if indoor_temp < 27 else C_RED)
    draw.text((14,  y + 2), f'{indoor_temp:.1f}°C',
              font=_font(36, bold=True), fill=t_col)

    pmv_col = (C_GREEN  if abs(pmv_val) <= 0.5 else
               C_ORANGE if abs(pmv_val) <= 1.5 else C_RED)
    draw.text((210, y +  4), f'PMV {pmv_val:+.2f}',
              font=_font(22, bold=True), fill=pmv_col)
    draw.text((210, y + 36), comfort_msg[:18],
              font=_font(17),            fill=pmv_col)

    # 사용자 선호 표시
    if pmv_preference > 0.0:
        pref_lbl, pref_col = f'따뜻하게 ({pmv_preference:+.1f})', C_HEAT
    elif pmv_preference < 0.0:
        pref_lbl, pref_col = f'시원하게 ({pmv_preference:+.1f})', C_COOL
    else:
        pref_lbl, pref_col = '중립 (±0.0)', C_LABEL

    draw.text(( 14, y + 70), '선호 설정:', font=_font(16), fill=C_LABEL)
    draw.text((110, y + 68), pref_lbl,     font=_font(18, bold=True), fill=pref_col)
    draw.text((440, y + 70), 'U:따뜻  D:시원', font=_font(14), fill=C_LABEL)
    y += 110

    # ── 2. 공기질 5단계 ──────────────────────────────────────────────────────
    pm10 = int(ds.get('pm10', 0))
    pm25 = int(ds.get('pm25', 0))
    khai = ds.get('khai', 0)

    pm10_lv, pm10_c = _aq_pm10(pm10)
    pm25_lv, pm25_c = _aq_pm25(pm25)
    khai_lv, khai_c = _aq_khai(khai)

    draw.rectangle([(0, y), (PANEL_W, y + 118)], fill=BG_SECT)
    draw.rectangle([(0, y), (PANEL_W, y + 38)],  fill=BG_HDR)
    draw.text((14, y + 8), '  공기질 현황 (5단계)',
              font=_font(19, bold=True), fill=C_TITLE)
    y += 38

    col_w = PANEL_W // 3
    for i, (lbl, val_str, lv, c) in enumerate([
        ('PM10',    f'{pm10} ㎍/㎥', pm10_lv, pm10_c),
        ('PM2.5',   f'{pm25} ㎍/㎥', pm25_lv, pm25_c),
        ('통합지수', f'KHAI {khai}',  khai_lv, khai_c),
    ]):
        cx = i * col_w + 16
        draw.text((cx, y +  2), lbl,     font=_font(14), fill=C_LABEL)
        draw.text((cx, y + 22), val_str, font=_font(15), fill=C_VAL)
        draw.text((cx, y + 46), lv,      font=_font(18, bold=True), fill=c)
    y += 80

    # ── 3. 창문 솔루션 메시지 ────────────────────────────────────────────────
    people   = ds.get('people_count', 0)
    win_msg, win_col = _window_advice(hvac, out_temp, pm10, pmv_val, people)

    draw.rectangle([(0, y), (PANEL_W, y + 78)], fill=(20, 26, 44))
    draw.rectangle([(0, y), (PANEL_W, y + 38)], fill=(36, 44, 70))
    draw.text((14, y + 8), '  창문 권장',
              font=_font(19, bold=True), fill=C_TEAL)
    y += 38
    draw.text((14, y + 4), f'▶  {win_msg}', font=_font(16), fill=win_col)
    y += 40

    # ── 4. PMV 보정 이력 그래프 ──────────────────────────────────────────────
    g_sect_h = 172
    draw.rectangle([(0, y), (PANEL_W, y + g_sect_h)], fill=BG_SECT)
    draw.rectangle([(0, y), (PANEL_W, y + 38)],        fill=BG_HDR)
    draw.text((14, y + 8), '  PMV 이력 그래프 (최근 30회)',
              font=_font(19, bold=True), fill=C_TITLE)
    y += 38
    _draw_pmv_graph(draw, 10, y + 4, PANEL_W - 20, 124, pmv_history, pmv_preference)
    y += g_sect_h - 38

    # ── 5. 에너지 절약 카운터 ────────────────────────────────────────────────
    y = _draw_energy(draw, y, em_data)

    # ── 6. 재실 예측 ─────────────────────────────────────────────────────────
    y = _draw_occ_pred(draw, y, occ_pred)

    # ── 하단 키 안내 ─────────────────────────────────────────────────────────
    remaining = PANEL_H - y
    if remaining > 0:
        draw.rectangle([(0, y), (PANEL_W, PANEL_H)], fill=(16, 20, 38))
        draw.text((14, y + 8),
                  'Q: 종료   U: 선호온도 올리기   D: 선호온도 낮추기   S: 즉시 VLM 분석',
                  font=_font(13), fill=C_LABEL)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
