@echo off
REM ============================================================
REM  labelImg Windows 一键打包脚本
REM  用法: 双击运行,或在本目录打开 cmd 执行 build_exe.bat
REM  输出: dist\labelImg\  (labelImg.exe + _internal 依赖目录)
REM ============================================================
setlocal
cd /d "%~dp0.."

REM ---- 1. 准备虚拟环境 ----
if not exist build_venv (
    echo [1/5] 创建虚拟环境 build_venv ...
    python -m venv build_venv || goto :error
)
call build_venv\Scripts\activate.bat

REM ---- 2. 安装依赖 ----
echo [2/5] 安装依赖 (PyQt5 / onnxruntime / pyinstaller) ...
python -m pip install --upgrade pip
python -m pip install -r packaging\requirements-windows.txt || goto :error

REM ---- 3. PyInstaller 打包 ----
echo [3/5] 打包生成 exe ...
pyinstaller packaging\labelImg.spec --noconfirm --clean || goto :error

REM ---- 4. 更新 VC++ 运行库 (关键!) ----
REM onnxruntime 需要 VC++ 2022 (14.40+),而 Python 安装器自带的运行库偏旧,
REM 直接打包会在部分机器上报 "DLL 初始化例程失败"。
REM 这里安装官方 VC++ 2022 运行库,并把新版 DLL 覆盖进输出目录。
echo [4/5] 安装 VC++ 2022 运行库并更新到输出目录 ...
if not exist "%TEMP%\vc_redist.x64.exe" (
    powershell -Command "Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vc_redist.x64.exe'"
)
"%TEMP%\vc_redist.x64.exe" /install /quiet /norestart
copy /Y "%SystemRoot%\System32\vcruntime140.dll"        "dist\labelImg\_internal\" >nul
copy /Y "%SystemRoot%\System32\vcruntime140_1.dll"      "dist\labelImg\_internal\" >nul
copy /Y "%SystemRoot%\System32\msvcp140.dll"            "dist\labelImg\_internal\" >nul
copy /Y "%SystemRoot%\System32\msvcp140_1.dll"          "dist\labelImg\_internal\" >nul
copy /Y "%SystemRoot%\System32\msvcp140_2.dll"          "dist\labelImg\_internal\" >nul
copy /Y "%SystemRoot%\System32\msvcp140_atomic_wait.dll" "dist\labelImg\_internal\" >nul
copy /Y "%SystemRoot%\System32\msvcp140_codecvt_ids.dll" "dist\labelImg\_internal\" >nul
copy /Y "%SystemRoot%\System32\concrt140.dll"           "dist\labelImg\_internal\" >nul

REM ---- 5. 拷贝模型 (可选) ----
REM 把训练好的 best.onnx 放到项目根目录的 model\ 子目录,打包时自动放进 dist
if exist "model\best.onnx" (
    echo [5/5] 拷贝模型 best.onnx 到 dist\labelImg\ ...
    copy /Y "model\best.onnx" "dist\labelImg\best.onnx" >nul
) else (
    echo [5/5] 未发现 model\best.onnx,跳过模型拷贝。
    echo        请手动把 best.onnx 放到 dist\labelImg\ 下,或在程序里
    echo        "AI 检测 -> AI 模型设置" 中指定模型路径。
)

echo.
echo 打包完成! 输出目录: dist\labelImg\
echo 分发时把整个 dist\labelImg 文件夹压缩即可。
pause
exit /b 0

:error
echo.
echo 打包失败,请检查上方错误信息。
pause
exit /b 1
