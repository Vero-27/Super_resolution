bash:Super_resolution/build/android/resources/container_build.sh
#!/bin/bash
set -e

PROJECT_ROOT="/home/user/hostproject"
BUILD_WORK_DIR="/home/user/build_work"
OUTPUT_DIR="${PROJECT_ROOT}/Output/android"

rm -rf "$BUILD_WORK_DIR"
mkdir -p "$BUILD_WORK_DIR"

if [ ! -f "$PROJECT_ROOT/main.py" ]; then
    echo "ERREUR: Fichier $PROJECT_ROOT/main.py introuvable."
    exit 1
fi

cp "$PROJECT_ROOT/main.py" "$BUILD_WORK_DIR/"
[ -d "$PROJECT_ROOT/Core" ] && cp -r "$PROJECT_ROOT/Core" "$BUILD_WORK_DIR/"
[ -d "$PROJECT_ROOT/GUI" ] && cp -r "$PROJECT_ROOT/GUI" "$BUILD_WORK_DIR/"
[ -d "$PROJECT_ROOT/fonts" ] && cp -r "$PROJECT_ROOT/fonts" "$BUILD_WORK_DIR/"


cp "$PROJECT_ROOT/build/android/resources/buildozer.spec" "$BUILD_WORK_DIR/"
cd "$BUILD_WORK_DIR"
yes | buildozer android debug

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*.apk
cp bin/*.apk "$OUTPUT_DIR"/

