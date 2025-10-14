param(
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

$APP_NAME = "Super_resolution"
$PROJECT_ROOT = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

if (-not (Test-Path (Join-Path $PROJECT_ROOT "main.py"))) {
    Write-Host "ERREUR: main.py introuvable. Racine detectee: $PROJECT_ROOT" -ForegroundColor Red
    exit 1
}

$CACHE_DIR = Join-Path $PROJECT_ROOT "build\windows\ressources\cache"
$PYTHON_DIR = Join-Path $CACHE_DIR "python"
$TEMP_DIR = Join-Path $PROJECT_ROOT "build\temp"
$OUTPUT_DIR = Join-Path $PROJECT_ROOT "output\windows"
$REQUIREMENTS = Join-Path $PROJECT_ROOT "requirements.txt"
$PYTHON_VERSION = "3.11.9"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  $APP_NAME - Windows EXE Builder" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

if ($Clean) {
    Write-Host "[CLEAN] Suppression du cache et des anciennes sorties..." -ForegroundColor Yellow
    Remove-Item -Path $CACHE_DIR, $OUTPUT_DIR -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "        Nettoyage termine." -ForegroundColor Green
}

New-Item -ItemType Directory -Path $CACHE_DIR, $OUTPUT_DIR, $TEMP_DIR -Force | Out-Null

$pythonExe = Join-Path $PYTHON_DIR "python.exe"
$pipExe = Join-Path $PYTHON_DIR "Scripts\pip.exe"

if (Test-Path $pythonExe) {
    Write-Host "[1/4] Python portable trouve dans le cache." -ForegroundColor Green
} else {
    Write-Host "[1/4] Installation de Python $PYTHON_VERSION..." -ForegroundColor Yellow
    $pythonZip = Join-Path $CACHE_DIR "python.zip"
    $pythonEmbedUrl = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-embed-amd64.zip"
    try {
        Invoke-WebRequest -Uri $pythonEmbedUrl -OutFile $pythonZip -UseBasicParsing
        Expand-Archive -Path $pythonZip -DestinationPath $PYTHON_DIR -Force
        Remove-Item $pythonZip
        $pthFile = Get-ChildItem -Path $PYTHON_DIR -Filter "*._pth" | Select-Object -First 1
        if ($pthFile) { (Get-Content $pthFile.FullName) -replace '#import site', 'import site' | Set-Content $pthFile.FullName }
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile (Join-Path $CACHE_DIR "get-pip.py") -UseBasicParsing
        & $pythonExe (Join-Path $CACHE_DIR "get-pip.py") --no-warn-script-location | Out-Null
        Write-Host "      Python installe." -ForegroundColor Green
    } catch {
        Write-Host "      Erreur installation Python: $_" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[2/4] Verification des dependances..." -ForegroundColor Yellow
$requirementsHashFile = Join-Path $CACHE_DIR "requirements.hash"
$currentHash = (Get-FileHash -Path $REQUIREMENTS -Algorithm MD5).Hash
if ((Test-Path $requirementsHashFile) -and ((Get-Content $requirementsHashFile) -eq $currentHash)) {
    Write-Host "      Dependances a jour." -ForegroundColor Green
} else {
    Write-Host "      Installation des dependances..."
    & $pipExe install --upgrade pip -q
    & $pipExe install -r $REQUIREMENTS -q
    & $pipExe install pyinstaller -q
    Set-Content -Path $requirementsHashFile -Value $currentHash
    Write-Host "      Dependances installees." -ForegroundColor Green
}

Write-Host "[3/4] Preparation du build..." -ForegroundColor Yellow
Remove-Item -Path (Join-Path $OUTPUT_DIR "*"), (Join-Path $TEMP_DIR "*") -Recurse -Force -ErrorAction SilentlyContinue

$buildArgs = @(
    "main.py",
    "--noconfirm",
    "--windowed",
    "--name=$APP_NAME",
    "--distpath=$OUTPUT_DIR",
    "--workpath=$TEMP_DIR",
    "--hidden-import=plyer.platforms.win.filechooser"
)

$excluded_dirs = @('build', 'output', 'dist', 'venv', '.git', '.idea', '.vscode', '__pycache__')
Get-ChildItem -Path $PROJECT_ROOT -Directory | ForEach-Object {
    if ($_.Name.ToLower() -notin $excluded_dirs) {
        Write-Host "        Ajout du dossier: $($_.Name)"
        $buildArgs += "--add-data", "$($_.Name);$($_.Name)"
    }
}

Write-Host "[4/4] Lancement du build de l'executable..." -ForegroundColor Yellow
$pyinstallerExe = Join-Path $PYTHON_DIR "Scripts\pyinstaller.exe"

try {
    $process = Start-Process -FilePath $pyinstallerExe -ArgumentList $buildArgs -WorkingDirectory $PROJECT_ROOT -NoNewWindow -Wait -PassThru
} catch {
    Write-Host "      ERREUR: Echec du lancement de PyInstaller." -ForegroundColor Red
    exit 1
}

if ($process.ExitCode -ne 0) {
    Write-Host "      ERREUR: PyInstaller a echoue (code: $($process.ExitCode))." -ForegroundColor Red
    exit 1
}

$buildOutputFolder = Join-Path $OUTPUT_DIR $APP_NAME
if (Test-Path $buildOutputFolder) {
    Get-ChildItem -Path $buildOutputFolder | Move-Item -Destination $OUTPUT_DIR -Force
    Remove-Item -Path $buildOutputFolder -Force
}

Remove-Item -Path $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $PROJECT_ROOT "$APP_NAME.spec") -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  BUILD REUSSI !" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Application disponible dans: $OUTPUT_DIR" -ForegroundColor White