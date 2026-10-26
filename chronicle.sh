#!/usr/bin/env bash
# ==============================================================================
# Chronicle Management Control Script
# Usage:
#   ./chronicle.sh                         - Launch Chronicle dashboard
#   ./chronicle.sh stop                    - Stop running dashboard
#   ./chronicle.sh restart                 - Restart dashboard
#   ./chronicle.sh status                  - Check server status & health
#   ./chronicle.sh logs                    - Tail server log output
#   ./chronicle.sh test                    - Run fast test suite
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PID_FILE="${ROOT_DIR}/.chronicle.pid"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/chronicle.log"

DEFAULT_SOURCE="synthetic"
DEFAULT_PORT=8080

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Determine Python / chronicle executable
if [ -x "${ROOT_DIR}/.venv/bin/chronicle" ]; then
    CHRONICLE_BIN="${ROOT_DIR}/.venv/bin/chronicle"
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif command -v chronicle >/dev/null 2>&1; then
    CHRONICLE_BIN="$(command -v chronicle)"
    PYTHON_BIN="$(command -v python)"
else
    echo -e "${RED}Error: Virtual environment not found at .venv/ and 'chronicle' not in PATH.${NC}"
    echo "Please run: python -m venv .venv && source .venv/bin/activate && pip install -e '.[ui]'"
    exit 1
fi

mkdir -p "${LOG_DIR}"

is_running() {
    if [ -f "${PID_FILE}" ]; then
        local pid
        pid=$(cat "${PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

get_pid() {
    if [ -f "${PID_FILE}" ]; then
        cat "${PID_FILE}"
    fi
}

find_port_pid() {
    local port="${1:-8080}"
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti :"${port}" 2>/dev/null || true
    elif command -v fuser >/dev/null 2>&1; then
        fuser "${port}/tcp" 2>/dev/null || true
    else
        ss -tulpn 2>/dev/null | grep ":${port} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -n1 || true
    fi
}

start_server() {
    local source="${1:-$DEFAULT_SOURCE}"
    local port="${2:-$DEFAULT_PORT}"

    if is_running; then
        local pid
        pid=$(get_pid)
        echo -e "${YELLOW}Chronicle is already running (PID: ${pid}).${NC}"
        echo -e "Access dashboard at: ${CYAN}http://127.0.0.1:${port}${NC}"
        return 0
    fi

    # Check if port is occupied by another process
    local port_pid
    port_pid=$(find_port_pid "${port}")
    if [ -n "${port_pid}" ]; then
        echo -e "${RED}Port ${port} is currently in use by process ${port_pid}.${NC}"
        echo "Stop it first or pass a different port: ./chronicle.sh start ${source} 8081"
        return 1
    fi

    echo -e "${BOLD}${CYAN}Starting Chronicle dashboard...${NC}"
    echo -e "  Source:   ${GREEN}${source}${NC}"
    echo -e "  Port:     ${GREEN}${port}${NC}"
    echo -e "  Log file: ${BLUE}${LOG_FILE}${NC}"

    # Export deterministic embedder if running synthetic to guarantee zero downloads
    if [ "${source}" = "synthetic" ]; then
        export CHRONICLE_EMBEDDER_BACKEND="deterministic"
    fi

    nohup setsid "${CHRONICLE_BIN}" serve \
        --source "${source}" \
        --port "${port}" \
        --host "127.0.0.1" \
        >> "${LOG_FILE}" 2>&1 < /dev/null &

    local pid=$!
    disown "${pid}" 2>/dev/null || true
    echo "${pid}" > "${PID_FILE}"

    # Wait up to 5 seconds for the server to bind
    local timeout=10
    local ready=false
    while [ "${timeout}" -gt 0 ]; do
        if is_running; then
            if grep -q "Uvicorn running on" "${LOG_FILE}" 2>/dev/null || grep -q "NiceGUI ready" "${LOG_FILE}" 2>/dev/null; then
                ready=true
                break
            fi
            sleep 0.5
            timeout=$((timeout - 1))
        else
            break
        fi
    done

    if is_running; then
        echo -e "${GREEN}✓ Chronicle dashboard successfully started!${NC} (PID: ${pid})"
        echo -e "\n  ${BOLD}URL:${NC} ${CYAN}http://127.0.0.1:${port}${NC}\n"
        echo -e "Switch between videos (${CYAN}synthetic${NC}, ${CYAN}tabletop${NC}, ${CYAN}cook${NC}, ${CYAN}play${NC}, ${CYAN}walking${NC}) directly in the UI dropdown."
        echo -e "\nUseful commands:"
        echo -e "  ${YELLOW}./chronicle.sh stop${NC}    - Stop the server"
        echo -e "  ${YELLOW}./chronicle.sh status${NC}  - Check status"
        echo -e "  ${YELLOW}./chronicle.sh logs${NC}    - View live logs"
    else
        echo -e "${RED}✗ Failed to start server. Check logs:${NC}"
        tail -n 15 "${LOG_FILE}"
        rm -f "${PID_FILE}"
        return 1
    fi
}

stop_server() {
    local stopped=false
    local port="${1:-$DEFAULT_PORT}"

    if [ -f "${PID_FILE}" ]; then
        local pid
        pid=$(cat "${PID_FILE}")
        echo -e "${YELLOW}Stopping Chronicle (PID: ${pid})...${NC}"
        if kill -0 "${pid}" 2>/dev/null; then
            kill -15 "${pid}" 2>/dev/null || true
            for _ in {1..10}; do
                if ! kill -0 "${pid}" 2>/dev/null; then
                    stopped=true
                    break
                fi
                sleep 0.3
            done

            if ! ${stopped} && kill -0 "${pid}" 2>/dev/null; then
                echo -e "${YELLOW}Process did not terminate gracefully; forcing kill...${NC}"
                kill -9 "${pid}" 2>/dev/null || true
                stopped=true
            fi
        else
            stopped=true
        fi
        rm -f "${PID_FILE}"
    fi

    # Also clean up any lingering process on the target port
    local port_pid
    port_pid=$(find_port_pid "${port}")
    if [ -n "${port_pid}" ]; then
        echo -e "${YELLOW}Terminating lingering process on port ${port} (PID: ${port_pid})...${NC}"
        kill -15 "${port_pid}" 2>/dev/null || kill -9 "${port_pid}" 2>/dev/null || true
        stopped=true
    fi

    if ${stopped}; then
        echo -e "${GREEN}✓ Chronicle dashboard stopped.${NC}"
    else
        echo -e "${BLUE}Chronicle is not running.${NC}"
    fi
}

server_status() {
    if is_running; then
        local pid
        pid=$(get_pid)
        echo -e "${GREEN}● Chronicle dashboard is RUNNING${NC} (PID: ${pid})"
        echo -e "  URL:      ${CYAN}http://127.0.0.1:${DEFAULT_PORT}${NC}"
        echo -e "  Log file: ${BLUE}${LOG_FILE}${NC}"
        echo -e "\n${BOLD}Recent Log Output (last 8 lines):${NC}"
        tail -n 8 "${LOG_FILE}" 2>/dev/null || echo "No logs yet."
    else
        echo -e "${RED}○ Chronicle dashboard is STOPPED${NC}"
        rm -f "${PID_FILE}"
    fi
}

serve_foreground() {
    local source="${1:-$DEFAULT_SOURCE}"
    local port="${2:-$DEFAULT_PORT}"

    echo -e "${BOLD}${CYAN}Launching Chronicle dashboard in foreground...${NC}"
    echo -e "Access at: ${CYAN}http://127.0.0.1:${port}${NC} (Press Ctrl+C to stop)\n"

    exec "${CHRONICLE_BIN}" serve \
        --source "${source}" \
        --port "${port}" \
        --host "127.0.0.1"
}

show_logs() {
    if [ ! -f "${LOG_FILE}" ]; then
        echo "No log file found at ${LOG_FILE}."
        exit 0
    fi
    echo -e "${BOLD}Following logs from ${BLUE}${LOG_FILE}${NC} (Press Ctrl+C to exit):"
    tail -f "${LOG_FILE}"
}

run_tests() {
    echo -e "${BOLD}${CYAN}Running Chronicle automated test suite...${NC}\n"
    "${ROOT_DIR}/.venv/bin/pytest"
    echo -e "\n${BOLD}${CYAN}Running ruff linter check...${NC}"
    "${ROOT_DIR}/.venv/bin/ruff" check chronicle tests
    echo -e "\n${BOLD}${CYAN}Running GraphRAG evaluation suite...${NC}"
    "${PYTHON_BIN}" -m chronicle.semantic.evaluation
}

print_help() {
    echo -e "${BOLD}Project Chronicle - Service Control Script${NC}"
    echo
    echo -e "${BOLD}Usage:${NC} ./chronicle.sh <command> [options]"
    echo
    echo -e "${BOLD}Commands:${NC}"
    echo -e "  ${GREEN}start${NC} [source] [port]   Start the web dashboard in background"
    echo -e "                          Available sources: ${CYAN}synthetic${NC}, ${CYAN}tabletop${NC}, ${CYAN}cook${NC}, ${CYAN}play${NC}, ${CYAN}walking${NC}"
    echo -e "  ${GREEN}stop${NC} [port]             Stop the background dashboard"
    echo -e "  ${GREEN}restart${NC} [source] [port] Restart the background dashboard"
    echo -e "  ${GREEN}status${NC}                  Show server status, PID and recent logs"
    echo -e "  ${GREEN}logs${NC}                    Tail live log output (-f)"
    echo -e "  ${GREEN}serve${NC} [source] [port]   Run the web dashboard in foreground"
    echo -e "  ${GREEN}test${NC}                    Run full test & verification suite"
    echo -e "  ${GREEN}sources${NC}                 List all registered video sources"
    echo -e "  ${GREEN}help${NC}                    Show this help message"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ./chronicle.sh start                    # Start with synthetic fixture on port 8080"
    echo -e "  ./chronicle.sh start tabletop           # Start with HoloLens tabletop on port 8080"
    echo -e "  ./chronicle.sh start cook 8085          # Start with Aria cook clip on port 8085"
    echo -e "  ./chronicle.sh stop                     # Stop running server"
    echo -e "  ./chronicle.sh status                   # Check if running"
    echo -e "  ./chronicle.sh test                     # Run all 250+ tests & benchmarks"
}

# ------------------------------------------------------------------------------
# Command Dispatcher
# ------------------------------------------------------------------------------
ACTION="${1:-launch}"
shift || true

case "${ACTION}" in
    launch|start)
        start_server "${1:-$DEFAULT_SOURCE}" "${2:-$DEFAULT_PORT}"
        ;;
    stop)
        stop_server "${1:-$DEFAULT_PORT}"
        ;;
    restart)
        stop_server "${2:-$DEFAULT_PORT}"
        start_server "${1:-$DEFAULT_SOURCE}" "${2:-$DEFAULT_PORT}"
        ;;
    status)
        server_status
        ;;
    logs|log)
        show_logs
        ;;
    serve|run)
        serve_foreground "${1:-$DEFAULT_SOURCE}" "${2:-$DEFAULT_PORT}"
        ;;
    test|tests)
        run_tests
        ;;
    sources)
        "${CHRONICLE_BIN}" sources
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        echo -e "${RED}Unknown command: ${ACTION}${NC}\n"
        print_help
        exit 1
        ;;
esac
