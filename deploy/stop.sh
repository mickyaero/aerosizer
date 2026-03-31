#!/usr/bin/env bash
#
# AeroSizer – Clean Shutdown
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$SCRIPT_DIR/aerosizer.pid"
PORT=8080

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }

# ─── Stop Tailscale Funnel ───────────────────────────────────────────
info "Disabling Tailscale Funnel..."
tailscale funnel --bg off 2>/dev/null || \
    tailscale funnel off 2>/dev/null || true
ok "Funnel disabled"

# ─── Stop the app server ────────────────────────────────────────────
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        info "Stopping AeroSizer (PID $PID)..."
        kill "$PID"
        for i in $(seq 1 10); do
            kill -0 "$PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        ok "Server stopped"
    else
        info "Server not running (stale PID file)"
    fi
    rm -f "$PIDFILE"
else
    info "No PID file found – server may not be running"
    ORPHAN=$(lsof -iTCP:$PORT -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$ORPHAN" ]; then
        info "Found orphaned process on port $PORT (PID $ORPHAN), stopping..."
        kill "$ORPHAN" 2>/dev/null || true
    fi
fi

# ─── Stop caffeinate ─────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/caffeinate.pid" ]; then
    CAFF_PID=$(cat "$SCRIPT_DIR/caffeinate.pid")
    kill "$CAFF_PID" 2>/dev/null || true
    rm -f "$SCRIPT_DIR/caffeinate.pid"
    ok "Sleep prevention disabled"
fi

# ─── Remove firewall rules ──────────────────────────────────────────
info "Removing pf firewall rules..."
sudo pfctl -a aerosizer -F all 2>/dev/null || true
ok "Firewall anchor 'aerosizer' cleared"

echo ""
echo -e "${GREEN}AeroSizer stopped and all security measures reverted.${NC}"
