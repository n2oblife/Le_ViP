@echo off
for /d %%a in ("%programfiles%\Microsoft Visual Studio*") do (
for /f "tokens=3 delims=\" %%x in ("%%a") do echo %%x
)
pause >nul