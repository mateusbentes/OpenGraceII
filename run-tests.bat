@echo off
REM OpenGraceII Windows smoke: runs every example once and writes the
REM full output to log.txt (the same report the sandbox's runs produce).
REM The build paths: the native cargo build's target\debug\examples and
REM target\release\examples, the cross-build's plain examples folder.
setlocal enabledelayedexpansion
cd /d %~dp0
set LOG=%~dp0log.txt
echo OpenGraceII Windows smoke > %LOG%
echo machine: %COMPUTERNAME% >> %LOG%
echo user: %USERNAME% >> %LOG%

for %%E in (d3d12_smoke vk_smoke gl_smoke cube_smoke_d3d12 cube_lit_d3d12 cube_post_d3d12 cube_volume_d3d12 cube_deferred_d3d12) do (
    echo === %%E === >> %LOG% 2>&1
    echo === %%E ===
    if exist target\debug\examples\%%E.exe (
        target\debug\examples\%%E.exe >> %LOG% 2>&1
    ) else if exist target\release\examples\%%E.exe (
        target\release\examples\%%E.exe >> %LOG% 2>&1
    ) else if exist examples\%%E.exe (
        examples\%%E.exe >> %LOG% 2>&1
    ) else if exist %%E.exe (
        %%E.exe >> %LOG% 2>&1
    ) else (
        echo SKIP: %%E.exe not found >> %LOG% 2>&1
        echo SKIP: %%E.exe not found
    )
    echo exit code: !ERRORLEVEL! >> %LOG% 2>&1
)

echo === done === >> %LOG%
echo done. the log is at %LOG%
endlocal
