@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Installation RaGME_UP - PROP
echo ========================================
echo.

REM Aller au repertoire parent (racine du projet)
cd /d "%~dp0.."

REM Verifier que Python est installe
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH
    echo.
    echo Veuillez installer Python 3.8 ou superieur depuis https://www.python.org/downloads/
    echo N'oubliez pas de cocher "Add Python to PATH" lors de l'installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python detecte :
python --version
echo.

REM Verifier que pip est installe
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] pip n'est pas installe
    echo.
    echo Essayez d'installer pip avec : python -m ensurepip --upgrade
    echo.
    pause
    exit /b 1
)

echo [OK] pip detecte :
pip --version
echo.

REM Mettre a jour pip
echo [ETAPE 1/5] Mise a jour de pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ATTENTION] La mise a jour de pip a echoue, mais on continue...
)
echo.

REM Installer les dependances principales
echo [ETAPE 2/5] Installation des dependances principales...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERREUR] L'installation des dependances a echoue
    echo Verifiez votre connexion internet et reessayez
    echo.
    pause
    exit /b 1
)
echo.

REM Installer les dependances pour la GUI (CustomTkinter)
echo [ETAPE 3/5] Installation des dependances GUI...
pip install customtkinter pillow
if errorlevel 1 (
    echo.
    echo [ATTENTION] L'installation de CustomTkinter a echoue
    echo La GUI de gestion CSV pourrait ne pas fonctionner
    echo.
)
echo.

REM ========================================
REM DETECTION GPU ET MODE OFFLINE
REM ========================================
echo ========================================
echo Detection GPU pour mode OFFLINE
echo ========================================
echo.

REM Detecter si NVIDIA GPU est present via nvidia-smi
set "GPU_DETECTED=0"
set "CUDA_VERSION="

nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if not errorlevel 1 (
    set "GPU_DETECTED=1"
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu=name --format^=csv^,noheader 2^>nul') do (
        set "GPU_NAME=%%a"
    )
    for /f "tokens=2 delims=:" %%a in ('nvidia-smi --query-gpu=driver_version --format^=csv^,noheader 2^>nul') do (
        set "DRIVER_VERSION=%%a"
    )
    echo [OK] GPU NVIDIA detecte: !GPU_NAME!
)

if "!GPU_DETECTED!"=="0" (
    echo [INFO] Aucun GPU NVIDIA detecte - Mode CPU uniquement
    echo.
    goto :ask_offline
)

REM Demander si l'utilisateur veut installer le mode offline
:ask_offline
echo.
echo ========================================
echo Mode OFFLINE (modeles locaux)
echo ========================================
echo.
echo Le mode OFFLINE permet d'utiliser des modeles IA locaux
echo sans connexion internet (Mistral-7B, BGE-M3, etc.)
echo.
echo Voulez-vous installer le support mode OFFLINE ?
echo.

if "!GPU_DETECTED!"=="1" (
    echo [Recommande] Vous avez un GPU NVIDIA - acceleration CUDA disponible
) else (
    echo [Info] Pas de GPU detecte - le mode OFFLINE utilisera le CPU
    echo        Les performances seront reduites
)
echo.

set /p INSTALL_OFFLINE="Installer le mode OFFLINE ? (O/N) : "

if /i "!INSTALL_OFFLINE!"=="O" goto :install_offline
if /i "!INSTALL_OFFLINE!"=="Y" goto :install_offline
if /i "!INSTALL_OFFLINE!"=="OUI" goto :install_offline
if /i "!INSTALL_OFFLINE!"=="YES" goto :install_offline
goto :skip_offline

:install_offline
echo.
echo [ETAPE 4/5] Installation du mode OFFLINE...
echo.

REM Installer PyTorch avec CUDA si GPU detecte
REM IMPORTANT: torch>=2.6.0 requis pour corriger CVE-2025-32434
if "!GPU_DETECTED!"=="1" (
    echo Installation de PyTorch avec support CUDA...
    echo.
    echo NOTE: PyTorch 2.6+ requis pour corriger une vulnerabilite de securite
    echo.
    echo Quelle version de CUDA souhaitez-vous utiliser ?
    echo   1. CUDA 12.1 (stable, recommande)
    echo   2. CUDA 12.4 (derniere version)
    echo   3. CPU uniquement (pas d'acceleration GPU)
    echo.
    set /p CUDA_CHOICE="Votre choix (1/2/3) : "

    if "!CUDA_CHOICE!"=="1" (
        echo.
        echo Installation PyTorch avec CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if "!CUDA_CHOICE!"=="2" (
        echo.
        echo Installation PyTorch avec CUDA 12.4...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo.
        echo Installation PyTorch CPU uniquement...
        pip install torch torchvision torchaudio
    )
) else (
    echo Installation de PyTorch (CPU uniquement)...
    pip install torch torchvision torchaudio
)

if errorlevel 1 (
    echo.
    echo [ATTENTION] L'installation de PyTorch a echoue
    echo Vous pouvez reessayer manuellement plus tard
    echo.
) else (
    echo [OK] PyTorch installe avec succes
)
echo.

REM Installer les autres dependances offline
echo Installation des dependances Transformers...
pip install transformers>=4.40.0 accelerate>=0.24.0 sentencepiece>=0.1.99 safetensors>=0.4.0 tokenizers>=0.14.0
if errorlevel 1 (
    echo.
    echo [ATTENTION] L'installation de Transformers a echoue
    echo.
)
echo.

REM Installer python-pptx pour traitement PowerPoint
echo Installation de python-pptx...
pip install python-pptx>=0.6.21
echo.

REM Verifier l'installation CUDA
echo.
echo [ETAPE 5/5] Verification de l'installation CUDA...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Aucun\"}')"
echo.

goto :end_install

:skip_offline
echo.
echo [INFO] Installation du mode OFFLINE ignoree
echo        Vous pourrez l'installer plus tard avec : scripts\install_offline.bat
echo.

:end_install
echo ========================================
echo Installation terminee avec succes !
echo ========================================
echo.
echo Vous pouvez maintenant lancer l'application avec : scripts\launch.bat
echo.

REM Afficher les informations de configuration
echo ----------------------------------------
echo Configuration detectee :
echo ----------------------------------------
if "!GPU_DETECTED!"=="1" (
    echo   GPU: !GPU_NAME!
    echo   Mode OFFLINE: Disponible avec acceleration CUDA
) else (
    echo   GPU: Non detecte
    echo   Mode OFFLINE: Disponible en mode CPU
)
echo.
echo   Stockage primaire: N:\...\FAISS_DATABASE
echo   Stockage fallback: D:\FAISS_DATABASE
echo.
echo ----------------------------------------
echo.

pause
