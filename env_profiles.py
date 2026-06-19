from __future__ import annotations

"""
[환경 프로파일]
사용 공간 유형별 기본 파라미터 정의.

VLM이 분석에 성공하면 VLM 값이 항상 우선 적용됨.
프로파일 값은 VLM 미실행/실패 시 fallback 으로만 사용.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvProfile:
    key:               str
    name:              str
    desc:              str       # 한 줄 설명
    icon_color:        tuple     # 카드 포인트 색 (R,G,B)
    # 근무/운영 시간
    work_start:        int       # 운영 시작 시각 (시)
    work_end:          int       # 운영 종료 시각 (시)
    # 점심 감지
    lunch_enabled:     bool
    lunch_start:       int       # 점심 시작 시각 (시)
    lunch_end:         int       # 점심 종료 시각 (시)
    # 퇴근 맥락 감지
    departure_enabled: bool
    # VLM 실패 시 fallback MET
    # (fallback CLO는 계절이 아닌 외부온도 기반 — main._fallback_clo 참고)
    met_baseline:      float
    # 특징 설명 (UI 카드에 표시)
    features:          tuple


PROFILES: dict[str, EnvProfile] = {
    "office": EnvProfile(
        key="office",
        name="사무실",
        desc="일반 사무 환경",
        icon_color=(21, 101, 192),
        work_start=9,  work_end=18,
        lunch_enabled=True,  lunch_start=12, lunch_end=13,
        departure_enabled=True,
        met_baseline=1.2,
        features=("출퇴근 패턴 (9~18시)", "점심시간 감지 (12~13시)",
                  "퇴근 맥락 자동 절전", "착석 기반 MET 1.2"),
    ),
    "home": EnvProfile(
        key="home",
        name="가정",
        desc="주거 공간",
        icon_color=(198, 90, 28),
        work_start=0,  work_end=24,
        lunch_enabled=False, lunch_start=12, lunch_end=13,
        departure_enabled=False,
        met_baseline=1.0,
        features=("24시간 상시 운영", "자유로운 재실 패턴",
                  "퇴근 감지 비활성", "가벼운 착의 기준"),
    ),
    "gym": EnvProfile(
        key="gym",
        name="체육시설",
        desc="헬스장 · 스포츠센터",
        icon_color=(46, 125, 50),
        work_start=6,  work_end=22,
        lunch_enabled=False, lunch_start=12, lunch_end=13,
        departure_enabled=False,
        met_baseline=2.5,
        features=("높은 대사율 MET 2.5", "운동 복장 기준",
                  "점심/퇴근 감지 비활성", "수시 입퇴장 패턴"),
    ),
    "facility": EnvProfile(
        key="facility",
        name="부대시설",
        desc="군부대 · 공공시설",
        icon_color=(55, 71, 79),
        work_start=8,  work_end=18,
        lunch_enabled=True,  lunch_start=12, lunch_end=13,
        departure_enabled=True,
        met_baseline=1.5,
        features=("규칙적 일과 (8~18시)", "점심시간 감지 (12~13시)",
                  "퇴근 맥락 자동 절전", "활동적 MET 1.5"),
    ),
}
