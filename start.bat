@echo off
title BigBrother — Starting...
cd /d "%~dp0"

echo.
echo  Starting BigBrother...
echo  Dashboard will open automatically in your browser.
echo  Press Ctrl+C in this window to stop.
echo.

c:\python312\python.exe main.py

pause
