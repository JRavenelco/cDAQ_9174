@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHONUTF8=1"

echo ================================================
echo   Rugosimetro TMR200 - Interfaz Python
echo ================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta disponible en PATH.
    echo Instala Python y vuelve a intentar.
    echo.
    pause
    exit /b 1
)

echo Ejecutando interfaz...
echo.
python "%SCRIPT_DIR%caracterizacion_rugosimetro.py"

if errorlevel 1 (
    echo.
    echo La interfaz termino con error.
    echo Si faltan dependencias, instala:
    echo   pip install -r "%SCRIPT_DIR%requirements_rugosimetro.txt"
)

echo.
pause
