@echo off
rem NVCC wrapper: nvcc on Windows needs the MSVC x64 environment from vcvars64.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
nvcc %*
exit /b %ERRORLEVEL%
