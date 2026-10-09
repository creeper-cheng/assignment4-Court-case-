@echo off
rem ============================================================
rem  IMPORTANT: keep this file PURE ASCII + CRLF.
rem  cmd.exe parses batch files by byte offset; a code-page change
rem  (chcp 65001) mid-file makes it mis-read multi-byte characters
rem  and truncate lines into bogus commands (window flashes and closes).
rem  All Chinese output comes from the Python side.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

rem ---- load judge.env (KEY=VALUE per line) ----
if not exist "judge.env" (
  echo [ERROR] judge.env not found. It must contain JUDGE_KEY and LLM_* settings.
  pause
  exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%a in ("judge.env") do (
  if not "%%a"=="" if not "%%a:~0,1%"=="#" set "%%a=%%b"
)

if not exist "_derived\kb.jsonl" (
  echo [ERROR] knowledge base not built. Run:  python build_kb.py
  pause
  exit /b 1
)

echo Starting judge API on http://127.0.0.1:8000 ...
echo Press Ctrl+C to stop.
python "serve.py"
pause
