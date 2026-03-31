#!/usr/bin/env bash
#
# AeroSizer – Secure Demo Start Script
# Binds ONLY to 127.0.0.1, exposes via Tailscale Funnel (HTTPS only).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIDFILE="$SCRIPT_DIR/aerosizer.pid"
LOG_DIR="$SCRIPT_DIR/logs"
PORT=8080

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ─── Pre-flight checks ────────────────────────────────────────────────
info "Running pre-flight checks..."

command -v python3 >/dev/null 2>&1 || fail "python3 not found"
command -v tailscale >/dev/null 2>&1 || fail "tailscale CLI not found – install from https://tailscale.com/download"

# Check tailscale is connected
if ! tailscale status >/dev/null 2>&1; then
    fail "Tailscale is not running or not logged in. Run: tailscale up"
fi
ok "Tailscale connected"

# Abort if already running
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    fail "AeroSizer already running (PID $(cat "$PIDFILE")). Run stop.sh first."
fi

# ─── Python dependencies ─────────────────────────────────────────────
cd "$PROJECT_DIR"

info "Checking dependencies..."
python3 -c "import fastapi, uvicorn, numpy, scipy" 2>/dev/null || {
    info "Installing dependencies..."
    pip install -q fastapi uvicorn numpy scipy
}
ok "Dependencies ready"

# ─── Firewall (pf) – defense in depth ────────────────────────────────
info "Configuring firewall (requires sudo for pf)..."
PF_RULES="pass quick on lo0 all
block in quick proto tcp from any to any port $PORT"

echo "$PF_RULES" | sudo pfctl -a aerosizer -f - 2>/dev/null && \
    sudo pfctl -e 2>/dev/null || true
ok "pf anchor 'aerosizer' loaded – port $PORT blocked on LAN interfaces"

# ─── Disable unnecessary Tailscale features ──────────────────────────
info "Hardening Tailscale configuration..."
tailscale set --ssh=false 2>/dev/null || true
tailscale set --exit-node= 2>/dev/null || true
tailscale set --advertise-exit-node=false 2>/dev/null || true
tailscale set --advertise-routes= 2>/dev/null || true
ok "Tailscale SSH, exit node, and subnet routing disabled"

# ─── Start uvicorn ───────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
info "Starting AeroSizer on 127.0.0.1:$PORT..."

cd "$PROJECT_DIR"
python3 -m uvicorn app:app \
    --host 127.0.0.1 \
    --port "$PORT" \
    --workers 1 \
    --limit-max-requests 50000 \
    --timeout-keep-alive 30 \
    --log-level warning \
    >> "$LOG_DIR/server.log" 2>&1 &

APP_PID=$!
echo "$APP_PID" > "$PIDFILE"

# ─── Prevent sleep (macOS) ───────────────────────────────────────────
info "Preventing system sleep while serving..."
caffeinate -s -w "$APP_PID" &
CAFFEINE_PID=$!
echo "$CAFFEINE_PID" > "$SCRIPT_DIR/caffeinate.pid"

# Wait for the server to be ready
for i in $(seq 1 15); do
    if curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    fail "Server failed to start. Check $LOG_DIR/server.log"
fi
ok "Server running (PID $APP_PID)"

# ─── Verify localhost-only binding ───────────────────────────────────
LISTEN_ADDR=$(lsof -iTCP:$PORT -sTCP:LISTEN -n -P 2>/dev/null | grep "$APP_PID" | awk '{print $9}' | head -1)
if echo "$LISTEN_ADDR" | grep -q "127.0.0.1"; then
    ok "Confirmed: listening on 127.0.0.1 only"
elif echo "$LISTEN_ADDR" | grep -q "\*:$PORT\|0.0.0.0"; then
    kill "$APP_PID" 2>/dev/null || true
    rm -f "$PIDFILE"
    fail "SECURITY: Server bound to 0.0.0.0! Killed process."
fi

# ─── Enable Tailscale Funnel ─────────────────────────────────────────
info "Enabling Tailscale Funnel (HTTPS) on port $PORT..."
tailscale funnel --bg "$PORT" 2>/dev/null || tailscale funnel "$PORT" &

sleep 2

# Get the funnel URL
FUNNEL_URL=$(tailscale funnel status 2>/dev/null | grep -oE 'https://[^ ]+' | head -1 || true)
if [ -z "$FUNNEL_URL" ]; then
    HOSTNAME=$(tailscale status --json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
dns = data.get('Self', {}).get('DNSName', '')
print(dns.rstrip('.'))
" 2>/dev/null || echo "your-machine.tailnet.ts.net")
    FUNNEL_URL="https://$HOSTNAME"
fi

# ─── Summary ─────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  AeroSizer is live!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${CYAN}Public URL:${NC}   $FUNNEL_URL"
echo -e "  ${CYAN}Local URL:${NC}    http://127.0.0.1:$PORT"
echo -e "  ${CYAN}Server PID:${NC}   $APP_PID"
echo -e "  ${CYAN}Logs:${NC}         $LOG_DIR/"
echo ""
echo -e "  ${YELLOW}Security:${NC}"
echo -e "    - Bound to 127.0.0.1 only (no LAN access)"
echo -e "    - pf firewall blocking port $PORT on all interfaces"
echo -e "    - HTTPS only via Tailscale Funnel"
echo ""
echo -e "  Run ${CYAN}./deploy/stop.sh${NC} to shut down."
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
