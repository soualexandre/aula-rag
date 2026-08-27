@echo off
REM Sobe a demo em http://127.0.0.1:8000
setlocal
cd /d "%~dp0"

REM No Windows nao existe "python3": o launcher e "py", e o executavel e "python".
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo Python nao encontrado no PATH.
  echo Instale em https://www.python.org/downloads/ marcando "Add python.exe to PATH".
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Criando a venv...
  %PY% -m venv .venv || exit /b 1
)

".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || exit /b 1

if not defined PORT set "PORT=8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port %PORT%
