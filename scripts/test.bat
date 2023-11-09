@ECHO OFF
setlocal ENABLEEXTENSIONS
; 32-bit system:
set KEY_NAME="SOFTWARE\Microsoft\VisualStudio\9.0\Setup\VS"
; 64-bit system:
; set KEY_NAME="SOFTWARE\WOW6432Node\Microsoft\VisualStudio\9.0\Setup\VS"
set VALUE_NAME=ProductDir

FOR /F "usebackq skip=4 tokens=1-3" %%A IN (`REG QUERY %KEY_NAME% /v %VALUE_NAME% 2^>nul`) DO (
    set ValueValue=%%C
)

if defined ValueName (
    @echo Registry Value = %ValueValue%
) else (
    @echo %KEY_NAME%\%VALUE_NAME% not found.
)
pause