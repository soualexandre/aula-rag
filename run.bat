@echo off
REM Sobe a demo em http://127.0.0.1:8000
setlocal enabledelayedexpansion
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
  REM Venv criada a partir do Anaconda costuma quebrar o onnxruntime: as DLLs
  REM do Anaconda entram na frente das nativas do pacote.
  for /f "delims=" %%i in ('%PY% -c "import sys;print(sys.base_prefix)"') do set "PYBASE=%%i"
  echo !PYBASE!| findstr /i "conda" >nul && (
    echo.
    echo AVISO: o Python encontrado e o do Anaconda ^(!PYBASE!^).
    echo Se a instalacao falhar com "DLL load failed", instale o Python do
    echo python.org, apague a pasta .venv e rode este script de novo.
    echo.
  )
  echo Criando a venv...
  %PY% -m venv .venv || exit /b 1
)

".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || exit /b 1

REM O onnxruntime (motor do fastembed) e a peca que costuma falhar no Windows:
REM ele depende do runtime do Visual C++. Testa antes de subir o servidor, para
REM o erro aparecer como instrucao e nao como traceback no startup do uvicorn.
".venv\Scripts\python.exe" -c "import onnxruntime" >nul 2>nul
if errorlevel 1 (
  echo.
  echo O onnxruntime nao carregou. Tentando uma versao mais antiga, que roda
  echo em maquinas sem o Visual C++ Redistributable mais recente...
  ".venv\Scripts\python.exe" -m pip install -q "onnxruntime==1.19.2" >nul 2>nul
  ".venv\Scripts\python.exe" -c "import onnxruntime" >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Nao foi possivel carregar o ONNX Runtime nesta maquina. Faca, nesta ordem:
    echo.
    echo   1^) Instale o Microsoft Visual C++ Redistributable ^(x64^) e reabra o terminal:
    echo      https://aka.ms/vs17/release/vc_redist.x64.exe
    echo.
    echo   2^) Se a venv veio do Anaconda, recrie com o Python do python.org:
    echo      rmdir /s /q .venv
    echo      py -3 -m venv .venv
    echo      run.bat
    echo.
    echo   3^) Rode o diagnostico para ver o erro completo:
    echo      .venv\Scripts\python -c "import onnxruntime"
    echo.
    exit /b 1
  )
  echo onnxruntime 1.19.2 instalado com sucesso.
)

if not defined PORT set "PORT=8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port %PORT%
