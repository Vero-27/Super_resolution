param(
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

$APP_NAME = "Super_resolution"
$PROJECT_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

if (-not (Test-Path (Join-Path $PROJECT_ROOT "main.py"))) {
    exit 1
}

$CACHE_DIR = Join-Path $PROJECT_ROOT "build\windows\resources\cache"
$PYTHON_DIR = Join-Path $CACHE_DIR "python"
$TEMP_DIR = Join-Path $PROJECT_ROOT "build\temp"
$OUTPUT_DIR = Join-Path $PROJECT_ROOT "output\windows"
$REQUIREMENTS = Join-Path $PROJECT_ROOT "requirements.txt"
$PYTHON_VERSION = "3.11.9"

if ($Clean) {
    Remove-Item -Path $CACHE_DIR, $OUTPUT_DIR -Recurse -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $CACHE_DIR, $OUTPUT_DIR, $TEMP_DIR -Force | Out-Null

$pythonExe = Join-Path $PYTHON_DIR "python.exe"
$pipExe = Join-Path $PYTHON_DIR "Scripts\pip.exe"
$sitePackagesPath = Join-Path $PYTHON_DIR "Lib\site-packages"

if (-not (Test-Path $pythonExe)) {

    $pythonInstaller = Join-Path $CACHE_DIR "python-installer.exe"
    $pythonInstallerUrl = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-amd64.exe"

    try {
        Write-Host "Téléchargement du programme d'installation complet de Python..."
        Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $pythonInstaller -UseBasicParsing

        Write-Host "Installation de Python dans $PYTHON_DIR (cela peut prendre un moment)..."
        $installArgs = @(
            "/quiet",
            "InstallAllUsers=0",
            "TargetDir=""$PYTHON_DIR""",
            "PrependPath=0",
            "Include_pip=1",
            "Include_tcltk=0",
            "Include_test=0",
            "Include_dev=0"
        )

        $process = Start-Process -FilePath $pythonInstaller -ArgumentList $installArgs -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            Write-Error "L'installation de Python a échoué avec le code $($process.ExitCode)"
            exit 1
        }

        Remove-Item $pythonInstaller
    } catch {
        Write-Error "Échec du téléchargement ou de l'installation de Python : $_"
        exit 1
    }
}

$requirementsHashFile = Join-Path $CACHE_DIR "requirements.hash"
$currentHash = (Get-FileHash -Path $REQUIREMENTS -Algorithm MD5).Hash
if (-not ((Test-Path $requirementsHashFile) -and ((Get-Content $requirementsHashFile) -eq $currentHash))) {

    & $pipExe install --upgrade pip -q
    & $pipExe install pyinstaller-hooks-contrib -q

    $tempRequirements = Join-Path $TEMP_DIR "temp_requirements.txt"
    Get-Content $REQUIREMENTS | Where-Object { $_ -notmatch '^(torch|torchvision)$' } | Set-Content $tempRequirements

    & $pipExe install -r $tempRequirements -q

    Write-Host "Installation de la version CPU de Torch et Torchvision..."
    & $pipExe install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

    & $pipExe install pyinstaller -q
    Set-Content -Path $requirementsHashFile -Value $currentHash
}

Remove-Item -Path (Join-Path $OUTPUT_DIR "*"), (Join-Path $TEMP_DIR "*") -Recurse -Force -ErrorAction SilentlyContinue

$buildArgs = @(
    "main.py",
    "--noconfirm",
    "--windowed",
    "--onefile",
    "--name=$APP_NAME",
    "--distpath=$OUTPUT_DIR",
    "--workpath=$TEMP_DIR",
    "--paths", $sitePackagesPath,
    "--hidden-import=plyer.platforms.win.filechooser",
    "--hidden-import=Core.models.espcn",
    "--hidden-import=Core.models.srcnn",
    "--hidden-import=Core.models.edsr_lite",
    "--collect-all", "torch",
    "--collect-all", "torchvision",
    "--collect-all", "cv2",
    "--collect-all", "numpy",
    "--collect-all", "PIL"
)

$buildArgs += "--add-data", "GUI;GUI"
$buildArgs += "--add-data", "fonts;fonts"
$buildArgs += "--add-data", "Core/checkpoints;Core/checkpoints"

$pyinstallerExe = Join-Path $PYTHON_DIR "Scripts\pyinstaller.exe"

try {
    Write-Host "--- Lancement de PyInstaller (ça va être long à cause de torch) ---"
    $process = Start-Process -FilePath $pyinstallerExe -ArgumentList $buildArgs -WorkingDirectory $PROJECT_ROOT -NoNewWindow -Wait -PassThru
} catch {
    exit 1
}

if ($process.ExitCode -ne 0) {
    exit 1
}

Remove-Item -Path $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $PROJECT_ROOT "$APP_NAME.spec") -ErrorAction SilentlyContinue