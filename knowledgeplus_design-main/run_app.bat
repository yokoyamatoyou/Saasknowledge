@echo off
REM Launch the unified Streamlit interface
cd /d "%~dp0"

IF EXIST "..\.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("..\.env") do set "%%A=%%B"
)

IF "%OPENAI_API_KEY%"=="" (
  ECHO OPENAI_API_KEY environment variable not set
  ECHO Add it to ..\.env or set it before running the app
  EXIT /B 1
)

streamlit run app.py

