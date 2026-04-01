#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load local env file if present (do not commit secrets to git).
if [[ -f "$ROOT_DIR/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.local"
  set +a
fi

# Override with env vars if needed.
SCENE_PYTHON="${SCENE_PYTHON:-python}"
SCENE_HOST="${SCENE_HOST:-127.0.0.1}"
SCENE_PORT="${SCENE_PORT:-8001}"
WEB_PORT="${WEB_PORT:-8000}"
SCENE_MODE="${SCENE_MODE:-headless}" # headless | windowed
SCENE_HEALTH_TIMEOUT_SECONDS="${SCENE_HEALTH_TIMEOUT_SECONDS:-180}"

if ! command -v "$SCENE_PYTHON" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $SCENE_PYTHON" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[WARN] OPENAI_API_KEY is not set. OpenAI-dependent routes may fail."
fi

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/runtime"
SCENE_LOG="${SCENE_LOG:-$ROOT_DIR/logs/scene_service.log}"
echo "[INFO] Starting scene service on ${SCENE_HOST}:${SCENE_PORT} (${SCENE_MODE})"
if [[ "$SCENE_MODE" == "windowed" ]]; then
  "$SCENE_PYTHON" -m app.backend.services.scene_service --host "$SCENE_HOST" --port "$SCENE_PORT" --windowed >"$SCENE_LOG" 2>&1 &
else
  "$SCENE_PYTHON" -m app.backend.services.scene_service --host "$SCENE_HOST" --port "$SCENE_PORT" --headless >"$SCENE_LOG" 2>&1 &
fi
SCENE_PID=$!

cleanup() {
  if kill -0 "$SCENE_PID" 2>/dev/null; then
    echo "[INFO] Stopping scene service (pid=$SCENE_PID)"
    kill "$SCENE_PID" || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Waiting for scene service health check..."
for ((i = 0; i < SCENE_HEALTH_TIMEOUT_SECONDS; i++)); do
  if ! kill -0 "$SCENE_PID" 2>/dev/null; then
    echo "[ERROR] Scene service exited before becoming healthy. Last log lines:" >&2
    tail -n 80 "$SCENE_LOG" >&2 || true
    exit 1
  fi
  if curl -fsS "http://${SCENE_HOST}:${SCENE_PORT}/health" >/dev/null 2>&1; then
    echo "[INFO] Scene service is ready."
    break
  fi
  sleep 1
done

if ! curl -fsS "http://${SCENE_HOST}:${SCENE_PORT}/health" >/dev/null 2>&1; then
  echo "[ERROR] Timed out waiting for scene service health check." >&2
  tail -n 80 "$SCENE_LOG" >&2 || true
  exit 1
fi

echo "[INFO] Starting Flask server on 0.0.0.0:${WEB_PORT}"
PORT="$WEB_PORT" python -m app.backend.app
