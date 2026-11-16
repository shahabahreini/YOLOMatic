#!/bin/sh
# Render the README panels in .github/readme/*.html to docs/assets/readme/*.png
# (2x, transparent corners). Run from the repository root. Needs Google Chrome
# and internet access for the web fonts.
set -e
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
SRC=.github/readme
OUT=docs/assets/readme
mkdir -p "$OUT"
render() { # name width height
  "$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --default-background-color=00000000 --virtual-time-budget=10000 \
    --window-size="$2,$3" --screenshot="$PWD/$OUT/$1.png" "file://$PWD/$SRC/$1.html" >/dev/null 2>&1
  echo "$OUT/$1.png"
}
render hero 1280 560
render stats 1280 170
render workflow 1280 440
render models 1280 450
