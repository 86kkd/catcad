#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="$ROOT_DIR/dist"
PLUGIN_DIR="$ROOT_DIR/kicad_plugin"
ZIP_NAME="rl_auto_router_plugin.zip"

mkdir -p "$OUT_DIR"
cd "$PLUGIN_DIR"
zip -r "$OUT_DIR/$ZIP_NAME" . -x "*.pyc" "__pycache__/" >/dev/null
echo "Plugin packaged: $OUT_DIR/$ZIP_NAME"
