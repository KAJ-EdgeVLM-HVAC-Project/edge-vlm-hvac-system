# 기여 가이드 (Contributing)

Edge VLM HVAC 프로젝트에 기여해 주셔서 감사합니다. 이 문서는 개발 환경 설정,
브랜치·커밋 규칙, Pull Request 절차를 정리합니다.

## 개발 환경

**Mac (개발):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_mac.txt
python main.py
```

**Jetson Orin Nano Super (배포):**
```bash
cd ~/edge-vlm-hvac-system
./run.sh                 # venv/DISPLAY 자동 감지, headless 자동 전환
```

자세한 아키텍처·모듈 설명은 [README.md](README.md)와 [CLAUDE.md](CLAUDE.md)를 참고하세요.

## 브랜치 전략 (GitHub Flow)

- `main` — 안정 버전. 직접 푸시 금지, PR을 통해서만 병합.
- `feature/<이름>` — 기능 개발 브랜치. 작업 후 `main`으로 PR.
- `fix/<이름>` — 버그 수정 브랜치.

```bash
git checkout -b feature/my-change
# ... 작업 ...
git push -u origin feature/my-change   # PR 생성
```

## 커밋 메시지 규칙

[Conventional Commits](https://www.conventionalcommits.org/) 형식을 권장합니다.

```
<type>: <한 줄 요약>

<본문 — 무엇을, 왜>
```

`type` 종류: `feat`(기능) · `fix`(버그) · `refactor`(리팩터링) ·
`perf`(성능) · `docs`(문서) · `style`(UI/포맷) · `test`(테스트) · `chore`(잡무)

예시: `fix: MPS OOM으로 VLM 스레드 사망 → clo 기본값 고정 문제 해결`

## Pull Request 절차

1. 이슈를 먼저 생성하거나 기존 이슈를 참조합니다.
2. `feature/` 또는 `fix/` 브랜치에서 작업합니다.
3. 푸시 전 로컬에서 확인:
   ```bash
   ruff check .            # 린트 (또는 flake8)
   pytest                  # 단위 테스트
   python -m py_compile main.py vlm_processor.py control_logic.py
   ```
4. PR을 생성하면 **CI 게이트**(린트 + 테스트)가 자동 실행됩니다. 통과해야 병합 가능합니다.
5. 최소 1인 코드 리뷰 후 `main`으로 병합합니다.

## 코드 스타일

- Python 3.8+ 호환 (Jetson은 3.10, Mac은 3.11). `from __future__ import annotations` 사용.
- 모듈은 단일 책임 원칙을 따릅니다 (예: `thermal_engine.py`는 PMV 계산만).
- 하드웨어 의존 코드(GPIO/CUDA)는 `try/except`로 감싸 다른 환경에서 깨지지 않게 합니다.

## 행동 강령

모든 기여자는 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)를 준수합니다.
