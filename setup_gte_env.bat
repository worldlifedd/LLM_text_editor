@echo off
setlocal

echo ============================================
echo   GTE plugin - Miniconda gte env setup
echo ============================================
echo.

set "CONDA_EXE=C:\Users\worldlife\miniconda3\Scripts\conda.exe"
set "GTE_PY=C:\Users\worldlife\miniconda3\envs\gte\python.exe"

if not exist "%CONDA_EXE%" (
    echo [ERROR] conda.exe not found: %CONDA_EXE%
    pause
    exit /b 1
)

echo [1/5] Accepting conda Terms of Service ...
echo.
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

echo.
echo [2/5] Creating conda env "gte" (Python 3.10) ...
echo.
"%CONDA_EXE%" create -n gte python=3.10 -y
if errorlevel 1 (
    echo.
    echo [ERROR] conda create failed
    pause
    exit /b 1
)

if not exist "%GTE_PY%" (
    echo.
    echo [ERROR] python.exe not found after env creation: %GTE_PY%
    pause
    exit /b 1
)

echo.
echo [3/5] Installing server deps (fastapi/uvicorn/pyyaml/requests) ...
echo.
"%GTE_PY%" -m pip install fastapi uvicorn pyyaml requests huggingface-hub pip-system-certs
if errorlevel 1 (
    echo.
    echo [ERROR] server deps install failed
    pause
    exit /b 1
)

echo.
echo [4/5] Installing llama-cpp-python (GGUF backend, AVX2 + CUDA 13 build) ...
echo     NOTE: official CUDA wheels (cu124/cu130) ship ggml-cpu.dll with AVX-512
echo     instructions, which crash (0xc000001d) on consumer CPUs (Intel 12/13/14th
echo     gen, Zen3). We use JamePeng's AVX2 build with runtime CPU-variant dispatch
echo     and bundled CUDA runtime instead.
echo.
"%GTE_PY%" -m pip install https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.48-cu130-win-20260821/llama_cpp_python-0.3.48%%2Bcu130-cp310-cp310-win_amd64.whl
if errorlevel 1 (
    echo.
    echo [WARN] llama-cpp-python install failed (API mode still works)
    echo       CUDA build: https://github.com/abetlen/llama-cpp-python#usage-with-gpu
)

echo.
echo [5/5] Verifying ...
echo.
"%GTE_PY%" --version
"%GTE_PY%" -c "import fastapi, uvicorn, yaml, requests; print('server deps OK')"
"%GTE_PY%" -c "import llama_cpp; print('llama-cpp-python OK, version:', llama_cpp.__version__)"
if errorlevel 1 (
    echo llama-cpp-python not installed or import failed ^(API mode still works^)
)

echo.
echo ============================================
echo   Done!
echo   Python: %GTE_PY%
echo   This path is set in .vscode/settings.json (gte.pythonCommand)
echo ============================================
echo.
pause
