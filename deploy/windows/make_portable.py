"""Windows 무설치 포터블 패키지 조립 — 리눅스에서 실행 가능.

Windows PC에 Python을 설치하거나 빌드할 필요 없이, 압축만 풀면 바로
실행되는 폴더를 만든다. PyInstaller는 크로스 컴파일이 안 되지만,
Python 공식 embeddable 배포판 + Windows 휠을 조합하면 같은 결과가 된다.

    python3 deploy/windows/make_portable.py [--out DIR] [--no-models]

결과 구조:
    SmartHVAC/
      SmartHVAC.bat        <- 더블클릭
      python/              embeddable Python + 라이브러리
      app/                 프로젝트 소스
      llama/               llama-server.exe
      models/              Qwen3-VL-2B GGUF
      yolo26s.onnx, demo.mp4, 사용법.txt
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

PY_VER = "3.12.10"
PY_URL = f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip"

# Windows에서 필요한 런타임 의존성만. torch/ultralytics/transformers는
# 넣지 않는다 - YOLO는 onnxruntime, VLM은 llama.cpp가 담당한다.
WHEELS = [
    "opencv-python<5",     # Jetson에서 검증한 4.x 계열로 고정
    "numpy<2",             # opencv 4.x가 빌드된 ABI에 맞춤
    "pandas<3",            # 3.0은 파괴적 변경 다수 - 검증한 2.x로 고정

    "scipy",               # 재실추적 헝가리안 매칭 (없으면 그리디로 품질 저하)
    "matplotlib",
    "pillow",
    "requests",
    "python-dotenv",
    "onnxruntime",
    "yt-dlp",
    # MSVC 런타임 DLL 일체. onnxruntime이 msvcp140_1.dll을, llama.cpp가
    # msvcp140.dll을 요구하는데 embeddable Python에는 없다. 이게 빠지면
    # "VCRUNTIME140_1.dll을 찾을 수 없습니다"로 실행이 중단된다.
    "msvc-runtime",
]

LLAMA_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"

LAUNCHER = """@echo off
chcp 65001 >nul 2>&1
title Smart HVAC - VLM 기반 지능형 공조 제어
cd /d "%~dp0"

REM 동봉된 llama.cpp / 모델을 쓰도록 지정
set "LCPP_HOME=%~dp0llama"
set "LCPP_MODELS=%~dp0models"
set "PYTHONPATH=%~dp0app"
set "PYTHONUNBUFFERED=1"
set "PYTHONUTF8=1"

echo.
echo   Smart HVAC 시작 중...
echo   (VLM 모델을 메모리에 올리는 데 10~30초 걸립니다)
echo.

"%~dp0python\\python.exe" "%~dp0app\\main.py" %*

if errorlevel 1 (
    echo.
    echo ============================================================
    echo  오류가 발생했습니다. 위 메시지를 확인해 주세요.
    echo ============================================================
    pause
)
"""


def log(msg):
    print(f"  {msg}", flush=True)


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "SmartHVAC-build"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r    {done * 100 / total:5.1f}% "
                                 f"({done / 1048576:.0f}/{total / 1048576:.0f} MB)")
                sys.stdout.flush()
    if total:
        print()
    os.replace(tmp, dest)


def stage_python(out, cache):
    """embeddable Python을 풀고 site-packages를 import 경로에 넣는다."""
    zip_path = os.path.join(cache, os.path.basename(PY_URL))
    if not os.path.exists(zip_path):
        log(f"Python {PY_VER} embeddable 다운로드")
        download(PY_URL, zip_path)
    pydir = os.path.join(out, "python")
    os.makedirs(pydir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(pydir)

    # embeddable 배포판은 기본적으로 site-packages를 보지 않는다.
    # ._pth 에 경로를 추가해야 설치한 라이브러리를 import할 수 있다.
    for name in os.listdir(pydir):
        if name.endswith("._pth"):
            p = os.path.join(pydir, name)
            lines = open(p).read().splitlines()
            # embeddable + ._pth 조합은 PYTHONPATH 환경변수를 무시한다.
            # 프로젝트 소스(app/)도 여기에 적어야 import가 된다.
            for entry in ("Lib\\site-packages", "..\\app"):
                if entry not in lines:
                    lines.insert(1, entry)
            lines = ["import site" if ln.strip() == "#import site" else ln
                     for ln in lines]
            open(p, "w").write("\n".join(lines) + "\n")
            log(f"{name} 수정 - site-packages 활성화")
    return pydir


def stage_wheels(pydir, cache):
    """Windows 휠을 받아 site-packages에 푼다 (pip 실행 없이 zip 해제)."""
    wdir = os.path.join(cache, "wheels")
    os.makedirs(wdir, exist_ok=True)
    if not os.listdir(wdir):
        log(f"Windows 휠 다운로드 ({len(WHEELS)}개 패키지 + 의존성)")
        cmd = [sys.executable, "-m", "pip", "download", "-q",
               "--only-binary=:all:", "--platform", "win_amd64",
               "--python-version", "3.12", "--dest", wdir] + WHEELS
        subprocess.run(cmd, check=True)

    site = os.path.join(pydir, "Lib", "site-packages")
    os.makedirs(site, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(wdir)):
        if not f.endswith(".whl"):
            continue
        with zipfile.ZipFile(os.path.join(wdir, f)) as z:
            z.extractall(site)
        n += 1

    # 휠 안에 섞여 나오는 잔여물 정리 (numpy 휠은 자기 자신을 포함한다).
    # *.data 는 콘솔 스크립트·헤더용이라 런타임에 불필요.
    for name in os.listdir(site):
        p = os.path.join(site, name)
        if name.endswith(".whl"):
            os.remove(p)
        elif name.endswith(".data") and os.path.isdir(p):
            shutil.rmtree(p)
    log(f"휠 {n}개 설치 -> {os.path.relpath(site, os.path.dirname(pydir))}")


def stage_llama(out, cache):
    """llama.cpp Windows x64 CPU 빌드를 llama/ 에 평평하게 푼다."""
    req = urllib.request.Request(LLAMA_API,
                                 headers={"User-Agent": "SmartHVAC-build"})
    with urllib.request.urlopen(req) as r:
        releases = json.load(r)

    def is_target(n):
        n = n.lower()
        return (n.startswith("llama-") and n.endswith(".zip")
                and "-bin-win-" in n and "cpu" in n and "x64" in n)

    pick = rel = None
    for c in releases:
        for a in c.get("assets", []):
            if is_target(a["name"]):
                rel, pick = c, a
                break
        if pick:
            break
    if pick is None:
        raise SystemExit("llama.cpp Windows x64 CPU 자산을 찾지 못했습니다")

    zip_path = os.path.join(cache, pick["name"])
    if not os.path.exists(zip_path):
        log(f"llama.cpp {rel['tag_name']} ({pick['size'] / 1048576:.0f} MB)")
        download(pick["browser_download_url"], zip_path)

    ldir = os.path.join(out, "llama")
    os.makedirs(ldir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            if m.endswith("/"):
                continue
            with z.open(m) as src, \
                 open(os.path.join(ldir, os.path.basename(m)), "wb") as dst:
                shutil.copyfileobj(src, dst)
    if not os.path.exists(os.path.join(ldir, "llama-server.exe")):
        raise SystemExit("zip 안에 llama-server.exe가 없습니다")

    log(f"llama.cpp 준비 ({rel['tag_name']})")


def stage_msvc_runtime(out, cache):
    """MSVC 런타임 DLL을 python/ 과 llama/ 양쪽에 배치.

    윈도우는 DLL을 "실행파일이 있는 폴더"에서 먼저 찾는다. python.exe와
    llama-server.exe는 서로 다른 폴더에 있으므로 양쪽 모두에 두어야
    VC++ 재배포 패키지 설치 없이 실행된다.
      - onnxruntime.dll  -> msvcp140_1.dll  (없으면 YOLO 검출 불가)
      - llama/*.dll      -> msvcp140.dll, vcruntime140*.dll
    """
    wdir = os.path.join(cache, "wheels")
    whl = next((f for f in os.listdir(wdir)
                if f.lower().startswith("msvc_runtime")), None)
    if whl is None:
        log("[경고] msvc-runtime 휠 없음 - 대상 PC에 VC++ 재배포 패키지 필요")
        return

    tmp = os.path.join(cache, "_msvc")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(os.path.join(wdir, whl)) as z:
        z.extractall(tmp)

    # 패키지가 실제로 요구하는 것만. vcamp140/vccorlib140 은 쓰지 않는데
    # D3DCOMPILER_47 등 추가 의존성을 끌고 들어와 오히려 검증을 어지럽힌다.
    NEEDED = {"msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
              "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
              "vcruntime140.dll", "vcruntime140_1.dll",
              "concrt140.dll", "vcomp140.dll"}
    dlls = {}
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.lower() in NEEDED:
                dlls.setdefault(f.lower(), os.path.join(root, f))

    for dest in (os.path.join(out, "python"), os.path.join(out, "llama")):
        os.makedirs(dest, exist_ok=True)
        for name, src in dlls.items():
            shutil.copy2(src, os.path.join(dest, name))
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"MSVC 런타임 DLL {len(dlls)}개 -> python/ 및 llama/")


def stage_app(out):
    """프로젝트 소스와 런타임 자산 복사."""
    app = os.path.join(out, "app")
    os.makedirs(app, exist_ok=True)
    srcs = subprocess.run(["git", "-C", REPO, "ls-files", "*.py"],
                          capture_output=True, text=True, check=True)
    n = 0
    for rel in srcs.stdout.split():
        if rel.startswith(("tests/", "deploy/")):
            continue
        shutil.copy2(os.path.join(REPO, rel), os.path.join(app, os.path.basename(rel)))
        n += 1
    log(f"소스 {n}개 복사")

    # yolo_detector.py 가 자기 파일 위치(_HERE) 기준으로 찾으므로 app/ 안에 둔다
    shutil.copy2(os.path.join(REPO, "yolo26s.onnx"), os.path.join(app, "yolo26s.onnx"))
    env = os.path.join(REPO, ".env.example")
    if os.path.exists(env):
        shutil.copy2(env, os.path.join(out, ".env"))
    guide = os.path.join(HERE, "사용법.txt")
    if os.path.exists(guide):
        shutil.copy2(guide, os.path.join(out, "사용법.txt"))

    demo = os.path.join(REPO, "docs", "demo", "vlm_hvac_demo.mp4")
    if os.path.exists(demo):
        shutil.copy2(demo, os.path.join(out, "demo.mp4"))
    else:  # 작업 트리에서 삭제된 경우 git에서 복원
        blob = subprocess.run(["git", "-C", REPO, "show",
                               "HEAD:docs/demo/vlm_hvac_demo.mp4"],
                              capture_output=True)
        if blob.returncode == 0 and blob.stdout:
            open(os.path.join(out, "demo.mp4"), "wb").write(blob.stdout)
    log("런타임 자산 복사 (onnx, demo.mp4, .env, 사용법)")


def stage_models(out):
    src = os.path.join(os.path.expanduser("~"), "llama.cpp", "models", "Qwen3-VL-2B")
    if not os.path.isdir(src):
        log(f"[건너뜀] 모델 폴더 없음: {src}")
        return
    dst = os.path.join(out, "models", "Qwen3-VL-2B")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if not f.endswith(".gguf"):
            continue
        d = os.path.join(dst, f)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(os.path.join(src, f)):
            continue
        log(f"모델 복사: {f} ({os.path.getsize(os.path.join(src, f)) / 1048576:.0f} MB)")
        shutil.copy2(os.path.join(src, f), d)


def main():
    p = argparse.ArgumentParser(description="Windows 무설치 패키지 조립")
    p.add_argument("--out", default=os.path.join(REPO, "dist", "SmartHVAC"))
    p.add_argument("--cache", default=os.path.join(REPO, "build_pyi", "cache"))
    p.add_argument("--no-models", action="store_true",
                   help="VLM 모델(1.9GB) 제외 - 구조만 확인할 때")
    args = p.parse_args()

    out, cache = os.path.abspath(args.out), os.path.abspath(args.cache)
    os.makedirs(cache, exist_ok=True)
    # 재빌드 시 이전 잔재가 남지 않도록 models/ 를 뺀 전부를 지운다
    # (models는 1.9GB라 매번 다시 복사하면 느리다)
    if os.path.isdir(out):
        for name in os.listdir(out):
            if name == "models":
                continue
            p = os.path.join(out, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(out, exist_ok=True)

    print(f"== Windows 무설치 패키지 조립 ==\n출력: {out}\n")
    pydir = stage_python(out, cache)
    stage_wheels(pydir, cache)
    stage_llama(out, cache)
    stage_msvc_runtime(out, cache)
    stage_app(out)
    if not args.no_models:
        stage_models(out)

    with open(os.path.join(out, "SmartHVAC.bat"), "w", newline="\r\n") as f:
        f.write(LAUNCHER)
    log("SmartHVAC.bat 생성")

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(out) for f in fs)
    print(f"\n완료: {out}  ({total / 1048576:.0f} MB)")
    print("Windows에서 SmartHVAC.bat 더블클릭으로 실행됩니다.")


if __name__ == "__main__":
    main()
