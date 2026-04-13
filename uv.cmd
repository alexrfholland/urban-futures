@echo off
setlocal
set "REPO_ROOT=%~dp0"
set "UV_EXE=%REPO_ROOT%.tools\uv\uv.exe"
if not exist "%UV_EXE%" (
  echo Repo-local uv not found at "%UV_EXE%".
  echo See FIRST_MACHINE_SETUP.md for bootstrap steps.
  exit /b 1
)
"%UV_EXE%" %*
