@echo off
REM Launch the unified Streamlit interface from repository root
IF "%OPENAI_API_KEY%"=="" (
  IF EXIST "%~dp0\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0\.env") do set "%%A=%%B"
  )
)

IF "%OPENAI_API_KEY%"=="" (
  ECHO OPENAI_API_KEY environment variable not set
  ECHO Add it to .env or set it before running the app
  EXIT /B 1
)
cd /d "%~dp0\knowledgeplus_design-main"
streamlit run app.py
