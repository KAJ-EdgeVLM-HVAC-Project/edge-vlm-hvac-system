#!/bin/bash
# 데스크톱 런처 설치 — 앱 목록 + 바탕화면에 아이콘 등록 (sudo 불필요)
set -e
PROJECT="$(cd "$(dirname "$0")/.." && pwd -P)"   # -P: 심볼릭 링크 대신 실제 경로

APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
sed "s#__PROJECT__#$PROJECT#g" "$PROJECT/deploy/hvac.desktop" > "$APPS/hvac.desktop"
chmod +x "$APPS/hvac.desktop"
update-desktop-database "$APPS" 2>/dev/null || true
echo "✅ 앱 목록에 등록: $APPS/hvac.desktop"

# 바탕화면 아이콘 (GNOME은 신뢰 표시가 있어야 더블클릭 실행됨)
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ]; then
    cp "$APPS/hvac.desktop" "$DESKTOP_DIR/hvac.desktop"
    chmod +x "$DESKTOP_DIR/hvac.desktop"
    gio set "$DESKTOP_DIR/hvac.desktop" metadata::trusted true 2>/dev/null || true
    echo "✅ 바탕화면에 추가: $DESKTOP_DIR/hvac.desktop"
else
    echo "ℹ️  바탕화면 폴더 없음 — 앱 목록에서 실행하세요"
fi

echo
echo "실행: 앱 목록에서 'Smart HVAC' 검색 또는 바탕화면 아이콘 더블클릭"
