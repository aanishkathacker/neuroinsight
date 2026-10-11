#!/bin/bash
# Capture portfolio screenshots with headless Chrome.
#   ./scripts/screenshots.sh
# Output: docs/screenshots/*.png  (committed to the repo)
#
# Each capture is wrapped in a hard timeout: headless Chrome can fail to exit when a
# page keeps fetching, because --virtual-time-budget never "completes" while network
# activity continues. The guard guarantees the script always terminates.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/screenshots"
PROFILE="$ROOT/.run/.chrome-shots"
CHROME="${NI_SHOT_CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
URL="${NI_SHOT_URL:-http://127.0.0.1:8000}"
LIMIT="${NI_SHOT_TIMEOUT:-45}"

[[ -x "$CHROME" ]] || { echo "ERROR: Chrome not found at $CHROME"; exit 1; }
curl -fsS -m 5 "$URL/api/health" >/dev/null || {
  echo "ERROR: app not reachable at $URL  (run: ./scripts/serve.sh start)"; exit 1; }

mkdir -p "$OUT"

shot() {  # $1 filename  $2 path  $3 height  $4 virtual-time ms
  local out="$OUT/$1"
  rm -f "$out"; rm -rf "$PROFILE"; mkdir -p "$PROFILE"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 --user-data-dir="$PROFILE" \
    --window-size=1440,"$3" --virtual-time-budget="$4" \
    --screenshot="$out" "$URL$2" >/dev/null 2>&1 &
  local pid=$!
  local t=0
  while kill -0 $pid 2>/dev/null && [ $t -lt "$LIMIT" ]; do sleep 1; t=$((t+1)); done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  pkill -f "user-data-dir=$PROFILE" 2>/dev/null

  if [[ -s "$out" ]]; then
    printf "  ok      %-22s %s\n" "$1" "$(du -h "$out" | cut -f1)"
  else
    printf "  FAILED  %-22s (timed out after %ss)\n" "$1" "$LIMIT"; return 1
  fi
}

echo "Capturing from $URL"
shot 01-upload.png   "/"                   1150 6000
shot 02-results.png  "/?sample=synthetic"  3450 12000
rm -rf "$PROFILE"
echo "Done -> $OUT"
ls -1 "$OUT"