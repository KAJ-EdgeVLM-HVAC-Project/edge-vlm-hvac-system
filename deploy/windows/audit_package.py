"""Windows 패키지 정적 검증 — 리눅스에서 실행.

윈도우 PC 없이도 "실행하면 DLL/모듈이 없다"류의 실패를 미리 잡는다.
  1) 모든 PE(.exe/.dll/.pyd)의 import DLL이 패키지 안이나 OS에 있는지
  2) 프로젝트 소스가 import하는 파이썬 모듈이 번들에 있는지
  3) ._pth 경로 설정이 올바른지

    python3 deploy/windows/audit_package.py dist/SmartHVAC
"""
import ast
import os
import sys

import pefile

# Windows 10/11이 기본 제공하는 DLL (설치 불필요)
SYSTEM_DLLS = {
    "kernel32.dll", "user32.dll", "advapi32.dll", "gdi32.dll", "shell32.dll",
    "ole32.dll", "oleaut32.dll", "ws2_32.dll", "ntdll.dll", "msvcrt.dll",
    "ucrtbase.dll", "crypt32.dll", "rpcrt4.dll", "secur32.dll", "shlwapi.dll",
    "comctl32.dll", "comdlg32.dll", "winmm.dll", "wldap32.dll", "version.dll",
    "bcrypt.dll", "ncrypt.dll", "userenv.dll", "iphlpapi.dll", "dbghelp.dll",
    "psapi.dll", "setupapi.dll", "cfgmgr32.dll", "powrprof.dll", "imm32.dll",
    "opengl32.dll", "glu32.dll", "dwmapi.dll", "uxtheme.dll", "d3d9.dll",
    "d3d11.dll", "dxgi.dll", "mf.dll", "mfplat.dll", "mfreadwrite.dll",
    "mfcore.dll", "avicap32.dll", "msvfw32.dll", "winhttp.dll", "wininet.dll",
    "normaliz.dll", "dnsapi.dll", "mswsock.dll", "netapi32.dll", "pdh.dll",
    "gdiplus.dll", "propsys.dll", "windowscodecs.dll", "dcomp.dll",
    "bcryptprimitives.dll", "kernelbase.dll", "combase.dll", "sechost.dll",
    "winspool.drv", "oleacc.dll", "wtsapi32.dll", "authz.dll", "mpr.dll",
    "wsock32.dll", "cabinet.dll", "msi.dll", "imagehlp.dll", "winsta.dll",
}


def is_system(name):
    n = name.lower()
    return (n in SYSTEM_DLLS
            or n.startswith("api-ms-win-")
            or n.startswith("ext-ms-win-")
            or n.startswith("python3"))     # python3.dll / python312.dll


def scan_pe(root):
    """패키지 내 PE 파일 목록과, 제공되는 DLL 이름 집합."""
    pes, provided = [], set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".exe", ".dll", ".pyd")):
                p = os.path.join(dirpath, f)
                pes.append(p)
                provided.add(f.lower())
    return pes, provided


def check_dlls(root):
    pes, provided = scan_pe(root)
    print(f"[1] PE 의존성 검사 — 대상 {len(pes)}개")
    missing = {}
    for p in pes:
        try:
            pe = pefile.PE(p, fast_load=True)
            pe.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        except Exception:
            continue
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
            dll = (entry.dll or b"").decode("ascii", "replace")
            if not dll:
                continue
            if dll.lower() in provided or is_system(dll):
                continue
            missing.setdefault(dll, []).append(os.path.relpath(p, root))
        pe.close()

    if not missing:
        print("    ✅ 누락 DLL 없음")
    else:
        for dll, users in sorted(missing.items()):
            print(f"    ❌ {dll}  <- {len(users)}개가 요구 (예: {users[0]})")
    return missing


def check_imports(root):
    """app/ 소스의 최상위 import가 번들에 있는지."""
    app = os.path.join(root, "app")
    site = os.path.join(root, "python", "Lib", "site-packages")
    stdlib_zip = os.path.join(root, "python", "python312.zip")
    print(f"\n[2] 파이썬 import 검사 — {app}")

    local = {f[:-3] for f in os.listdir(app) if f.endswith(".py")}
    bundled = set()
    if os.path.isdir(site):
        for n in os.listdir(site):
            bundled.add(n[:-3] if n.endswith(".py") else n.split("-")[0])
    stdlib = set()
    if os.path.exists(stdlib_zip):
        import zipfile
        with zipfile.ZipFile(stdlib_zip) as z:
            for n in z.namelist():
                stdlib.add(n.split("/")[0].removesuffix(".pyc").removesuffix(".py"))
    # pyd/dll 확장 모듈
    for d in (os.path.join(root, "python", "DLLs"), os.path.join(root, "python")):
        if os.path.isdir(d):
            for n in os.listdir(d):
                if n.endswith(".pyd"):
                    bundled.add(n.split(".")[0])

    known = local | bundled | stdlib | set(sys.builtin_module_names)
    missing = {}
    for f in sorted(os.listdir(app)):
        if not f.endswith(".py"):
            continue
        try:
            tree = ast.parse(open(os.path.join(app, f), encoding="utf-8").read())
        except SyntaxError as e:
            print(f"    ❌ {f} 문법 오류: {e}")
            continue
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                for a in node.names:
                    mod = a.name.split(".")[0]
                    if mod not in known:
                        missing.setdefault(mod, set()).add(f)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mod = node.module.split(".")[0]
                if mod not in known:
                    missing.setdefault(mod, set()).add(f)

    if not missing:
        print("    ✅ 누락 모듈 없음")
    else:
        for m, users in sorted(missing.items()):
            print(f"    ⚠️  {m}  <- {', '.join(sorted(users))}")
    return missing


def check_pth(root):
    print("\n[3] ._pth 경로 설정")
    pyd = os.path.join(root, "python")
    pths = [f for f in os.listdir(pyd) if f.endswith("._pth")]
    if not pths:
        print("    ❌ ._pth 파일 없음")
        return {"_pth": True}
    lines = open(os.path.join(pyd, pths[0])).read().splitlines()
    need = {"..\\app": "프로젝트 소스", "Lib\\site-packages": "설치 라이브러리"}
    bad = {}
    for entry, desc in need.items():
        if entry in lines:
            print(f"    ✅ {entry}  ({desc})")
        else:
            print(f"    ❌ {entry} 누락  ({desc})")
            bad[entry] = True
    return bad


def check_layout(root):
    print("\n[4] 필수 파일")
    need = ["SmartHVAC.bat", "python/python.exe", "app/main.py",
            "app/vlm_processor.py", "app/yolo26s.onnx", "llama/llama-server.exe"]
    bad = {}
    for n in need:
        ok = os.path.exists(os.path.join(root, n))
        print(f"    {'✅' if ok else '❌'} {n}")
        if not ok:
            bad[n] = True
    models = os.path.join(root, "models")
    ggufs = []
    for dp, _, fs in os.walk(models):
        ggufs += [f for f in fs if f.endswith(".gguf")]
    print(f"    {'✅' if len(ggufs) >= 2 else '❌'} models/*.gguf ({len(ggufs)}개)")
    if len(ggufs) < 2:
        bad["models"] = True
    return bad


if __name__ == "__main__":
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "dist/SmartHVAC")
    print(f"=== Windows 패키지 정적 검증: {root} ===\n")
    bad = bool(check_dlls(root)) | bool(check_pth(root)) | bool(check_layout(root))
    check_imports(root)          # 경고만 (동적 import 오탐 가능)
    print("\n" + ("❌ 문제 발견 — 위 항목을 고쳐야 합니다" if bad
                  else "✅ 정적 검증 통과"))
    sys.exit(1 if bad else 0)
