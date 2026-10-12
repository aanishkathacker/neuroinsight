#!/usr/bin/env bash
# Public HTTPS link to the local app, via a Cloudflare Quick Tunnel.
#
# WHAT THIS IS
# ------------
# A temporary public URL so you can demo NeuroInsight to someone who is not on your
# network. getUserMedia requires a secure context, so plain http://<lan-ip>:8000 will
# NOT work -- browsers only grant camera access over HTTPS or on localhost. The tunnel
# supplies HTTPS.
#
# WHAT THIS IS NOT
# ----------------
# It is not a deployment. The link dies when you stop this script, it changes every
# time you restart it, and it only works while this laptop is awake and on the same
# network. Do not put this URL in a portfolio.
#
# FOR A PERMANENT LINK: see render.yaml and the "Deploy" section of README.md.
#
# Usage:
#   ./scripts/tunnel.sh start|stop|status|url|logs

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/.run"
LOG="$RUN/tunnel.log"
URLF="$RUN/tunnel.url"
mkdir -p "$RUN"

CF="$(command -v cloudflared || echo /opt/homebrew/bin/cloudflared)"
PORT="${PORT:-8000}"

case "${1:-status}" in
  start)
    if pgrep -f "cloudflared tunnel --url http://127.0.0.1:$PORT" >/dev/null 2>&1; then
      echo "already running: $(cat "$URLF" 2>/dev/null)"
      exit 0
    fi
    [ -x "$CF" ] || { echo "cloudflared not found. Install: brew install cloudflared"; exit 1; }

    ./scripts/serve.sh start >/dev/null 2>&1
    : > "$LOG"
    # macOS has no setsid, so nohup + background is the portable equivalent.
    nohup "$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >> "$LOG" 2>&1 &
    echo $! > "$RUN/tunnel.pid"

    # Cold start is genuinely slow on a first connection; poll rather than sleep once.
    printf "waiting for the tunnel"
    for _ in $(seq 1 40); do
      url=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG" 2>/dev/null | head -1)
      if [ -n "$url" ]; then
        # A registered connection is not yet a reachable edge - confirm with a real
        # request, which is what caught the premature "connected" reading earlier.
        code=$(curl -s -o /dev/null -m 20 -w "%{http_code}" "$url/api/health" 2>/dev/null)
        if [ "$code" = "200" ]; then
          echo "$url" > "$URLF"
          echo ""
          echo "  public link : $url"
          echo "  health      : $url/api/health"
          exit 0
        fi
      fi
      printf "."
      sleep 3
    done
    echo ""
    echo "tunnel did not come up - see $LOG"
    tail -5 "$LOG" | sed 's/^/  /'
    exit 1
    ;;

  stop)
    pkill -f "cloudflared tunnel --url http://127.0.0.1:$PORT" 2>/dev/null && echo "stopped" || echo "not running"
    rm -f "$RUN/tunnel.pid"
    ;;

  url)
    if [ -f "$URLF" ]; then
      url=$(cat "$URLF")
      code=$(curl -s -o /dev/null -m 15 -w "%{http_code}" "$url/api/health" 2>/dev/null)
      if [ "$code" = "200" ]; then echo "$url  (live)"
      else echo "$url  (STALE - tunnel is down; run: ./scripts/tunnel.sh start)"; fi
    else
      echo "no url yet - run: ./scripts/tunnel.sh start"
    fi
    ;;

  logs) tail -"${2:-40}" "$LOG" 2>/dev/null | sed 's/^/  /' ;;

  status|*)
    if pgrep -f "cloudflared tunnel --url http://127.0.0.1:$PORT" >/dev/null 2>&1; then
      echo "tunnel: running"
      [ -f "$URLF" ] && echo "  $(./scripts/tunnel.sh url)"
    else
      echo "tunnel: not running"
    fi
    curl -s -m 5 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 \
      && echo "app   : running on :$PORT" || echo "app   : not running"
    ;;
esac