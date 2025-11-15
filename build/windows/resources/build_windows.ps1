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

if (-not (Test-Path $pythonExe)) {
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
    } catch {
        exit 1
    }
}

$requirementsHashFile = Join-Path $CACHE_DIR "requirements.hash"
$currentHash = (Get-FileHash -Path $REQUIREMENTS -Algorithm MD5).Hash
if (-not ((Test-Path $requirementsHashFile) -and ((Get-Content $requirementsHashFile) -eq $currentHash))) {
    & $pipExe install --upgrade pip -q
    & $pipExe install -r $REQUIREMENTS -q
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
    "--hidden-import=plyer.platforms.win.filechooser"
)

$excluded_dirs = @('build', 'output', 'dist', 'venv', '.git', '.idea', '.vscode', '__pycache__', '.buildozer', 'bin')
Get-ChildItem -Path $PROJECT_ROOT -Directory | ForEach-Object {
    if ($_.Name.ToLower() -notin $excluded_dirs) {
        $buildArgs += "--add-data", "$($_.Name);$($_.Name)"
    }
}

$pyinstallerExe = Join-Path $PYTHON_DIR "Scripts\pyinstaller.exe"

try {
    $process = Start-Process -FilePath $pyinstallerExe -ArgumentList $buildArgs -WorkingDirectory $PROJECT_ROOT -NoNewWindow -Wait -PassThru
} catch {
    exit 1
}

if ($process.ExitCode -ne 0) {
    exit 1
}

Remove-Item -Path $TEMP_DIR -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $PROJECT_ROOT "$APP_NAME.spec") -ErrorAction SilentlyContinue