@echo off
REM Build upstream llama.cpp's llama-cli with MSVC, so the reference binary actually runs on this machine.
REM The shipped build-ref was MinGW/UCRT and imports api-ms-win-crt-private-l1-1-0.dll, which is a private UCRT
REM shim absent on consumer Windows - it exits 127 with no output. MSVC links the public UCRT and runs.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo VCVARS FAILED & exit /b 1)
set SRC=C:\Users\eman5\WuBu\research\upstream-ggml
set BLD=%SRC%\build-msvc
cmake -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON
if errorlevel 1 (echo CONFIGURE FAILED & exit /b 2)
cmake --build "%BLD%" --target llama-app -j 6
if errorlevel 1 (echo BUILD FAILED & exit /b 3)
echo BUILD OK
dir "%BLD%\bin\llama-cli.exe"
