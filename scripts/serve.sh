#!/bin/bash
# Start / stop / status for the NeuroInsight web app.
#   ./scripts/serve.sh start | stop | restart | status | logs
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
HOST="${NI_HOST:-127.0.0.1}"
PORT="${NI_PORT:-8000}"
PIDFILE="$ROOT/.run/uvicorn.pid"
LOG="$ROOT/.run/uvicorn.log"
URL="http://$HOST:$PORT"

mkdir -p "$ROOT/.run"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

start() {
  if running; then
    echo "already running on $URL (pid $(cat "$PIDFILE"))"
    return 0
  fi
  # fail fast with a clear message instead of a silent bind error
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: port $PORT is already in use by another process."
    echo "       Either stop it, or run: NI_PORT=8001 $0 start"
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | tail -n +2
    return 1
  fi
  cd "$ROOT"
  nohup "$PY" -m uvicorn neuroinsight.web.app:app \
        --host "$HOST" --port "$PORT" --log-level warning \
        > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"

  for _ in $(seq 1 40); do
    sleep 0.25
    if curl -fsS -m 2 "$URL/api/health" >/dev/null 2>&1; then
      echo "NeuroInsight running at $URL  (pid $(cat "$PIDFILE"))"
      echo "  stop with: $0 stop"
      return 0
    fi
  done
  echo "ERROR: failed to become healthy. Last 20 log lines:"
  tail -20 "$LOG"
  return 1
}

stop() {
  if running; then
    kill "$(cat "$PIDFILE")" 2>/dev/null
    for _ in $(seq 1 20); do running || break; sleep 0.2; done
    running && kill -9 "$(cat "$PIDFILE")" 2>/dev/null
    echo "stopped"
  else
    echo "not running (no live pidfile)"
  fi
  rm -f "$PIDFILE"
}

status() {
  if running; then
    echo "running  pid $(cat "$PIDFILE"))  $URL"
  else
    echo "not running"
  fi
  curl -fsS -m 2 "$URL/api/health" 2>/dev/null && echo || true
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  logs)    tail -n "${2:-40}" "$LOG" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs}"; exit 2 ;;
esac