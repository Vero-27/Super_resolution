batch:Super_resolution/build/android/build_apk.bat
@echo off
SETLOCAL EnableDelayedExpansion

echo === Lancement du build Android via Docker ===

pushd %~dp0..\..\
SET PROJECT_ROOT=%CD%
popd


SET RESOURCES_DIR=%~dp0resources
SET CACHE_DIR=%RESOURCES_DIR%\.buildozer_cache

IF NOT EXIST "%CACHE_DIR%" mkdir "%CACHE_DIR%"

docker run --rm ^
    -v "%PROJECT_ROOT%:/home/user/hostproject" ^
    -v "%CACHE_DIR%:/home/user/.buildozer" ^
    --entrypoint /bin/bash ^
    kivy/buildozer ^
    "/home/user/hostproject/build/android/resources/container_build.sh"

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo === BUILD REUSSI ===
    echo Votre APK devrait etre dans : %PROJECT_ROOT%\Output\android\
) ELSE (
    echo.
    echo === ECHEC DU BUILD ===
    echo Verifiez les logs ci-dessus.
)

pause