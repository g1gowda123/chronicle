#!/usr/bin/env bash
# ==============================================================================
# Chronicle Management Control Script
# Usage:
#   ./chronicle.sh                         - Launch Chronicle dashboard
#   ./chronicle.sh --no-llm                - Launch with deterministic answers
#   ./chronicle.sh --llm ollama            - Force the local Qwen backend
#   ./chronicle.sh --llm-model qwen2.5:7b-instruct
#   ./chronicle.sh stop                    - Stop running dashboard
#   ./chronicle.sh restart                 - Restart dashboard
#   ./chronicle.sh status                  - Check server status & health
#   ./chronicle.sh logs                    - Tail server log output
#   ./chronicle.sh mission [source] [port] - Start the zine mission control (:8081)
#   ./chronicle.sh mission-stop            - Stop the zine mission control
#   ./chronicle.sh mission-logs            - Tail zine mission control logs
#   ./chronicle.sh test                    - Run fast test suite
#   ./chronicle.sh preflight               - PASS/WARN/FAIL demo readiness check
#   ./chronicle.sh preflight --warm-llm    - Also load Qwen before the demo
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PID_FILE="${ROOT_DIR}/.chronicle.pid"
PORT_FILE="${ROOT_DIR}/.chronicle.port"
LLM_FILE="${ROOT_DIR}/.chronicle.llm"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/chronicle.log"

# Zine mission control (dashboard/run_dashboard.py) uses its own pid/port files.
MISSION_PID_FILE="${ROOT_DIR}/.chronicle-ui.pid"
MISSION_PORT_FILE="${ROOT_DIR}/.chronicle-ui.port"
MISSION_LOG_FILE="${ROOT_DIR}/logs/mission.log"
MISSION_DEFAULT_PORT=8081

DEFAULT_SOURCE="synthetic"
DEFAULT_PORT=8080
LLM_BACKEND="${CHRONICLE_LLM_BACKEND:-auto}"
LLM_MODEL="${CHRONICLE_LLM_MODEL:-qwen2.5:7b-instruct}"
OLLAMA_URL="${CHRONICLE_OLLAMA_URL:-http://127.0.0.1:11434}"
LLM_ARGS=()
PARSE_POSITIONAL=()

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

# Split LLM flags from positional [source] [port] arguments.
parse_llm_flags() {
    PARSE_POSITIONAL=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --no-llm)
                LLM_ARGS=(--llm template)
                shift
                ;;
            --llm)
                if [ "$#" -lt 2 ]; then
                    echo "Error: --llm needs a backend (auto|ollama|transformers|template)" >&2
                    exit 2
                fi
                LLM_ARGS+=(--llm "$2")
                shift 2
                ;;
            --llm-model)
                if [ "$#" -lt 2 ]; then
                    echo "Error: --llm-model needs an Ollama model name" >&2
                    exit 2
                fi
                LLM_ARGS+=(--llm-model "$2")
                shift 2
                ;;
            *)
                PARSE_POSITIONAL+=("$1")
                shift
                ;;
        esac
    done
}

# Resolve the effective backend/model after flag parsing.
effective_llm() {
    local backend="${LLM_BACKEND}"
    local model="${LLM_MODEL}"
    local i=0
    local n=${#LLM_ARGS[@]}
    while [ "${i}" -lt "${n}" ]; do
        case "${LLM_ARGS[${i}]}" in
            --llm)
                backend="${LLM_ARGS[$((i + 1))]:-${backend}}"
                i=$((i + 2))
                ;;
            --llm-model)
                model="${LLM_ARGS[$((i + 1))]:-${model}}"
                i=$((i + 2))
                ;;
            *)
                i=$((i + 1))
                ;;
        esac
    done
    printf '%s|%s' "${backend}" "${model}"
}

llm_summary() {
    llm_summary_info "$(effective_llm)"
}

llm_summary_info() {
    local info="$1"
    local backend="${info%%|*}"
    local model="${info##*|}"
    if [ "${backend}" = "template" ]; then
        echo "LLM:      template (deterministic answers)"
    elif curl -sf --max-time 1 "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
        echo "LLM:      ollama / ${model} (server up at ${OLLAMA_URL})"
    else
        echo "LLM:      ollama / ${model} (server offline — will fall back to template)"
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
    echo -e "  $(llm_summary)"
    echo -e "  Log file: ${BLUE}${LOG_FILE}${NC}"

    # Export deterministic embedder if running synthetic to guarantee zero downloads
    if [ "${source}" = "synthetic" ]; then
        export CHRONICLE_EMBEDDER_BACKEND="deterministic"
    fi

    nohup setsid "${CHRONICLE_BIN}" serve \
        --source "${source}" \
        --port "${port}" \
        --host "127.0.0.1" \
        "${LLM_ARGS[@]}" \
        >> "${LOG_FILE}" 2>&1 < /dev/null &

    local pid=$!
    disown "${pid}" 2>/dev/null || true
    echo "${pid}" > "${PID_FILE}"
    echo "${port}" > "${PORT_FILE}"
    effective_llm > "${LLM_FILE}"

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
        rm -f "${PID_FILE}" "${PORT_FILE}" "${LLM_FILE}"
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
        local pid port info
        pid=$(get_pid)
        port="${DEFAULT_PORT}"
        [ -f "${PORT_FILE}" ] && port=$(cat "${PORT_FILE}")
        info="$(effective_llm)"
        [ -f "${LLM_FILE}" ] && info=$(cat "${LLM_FILE}")
        echo -e "${GREEN}● Chronicle dashboard is RUNNING${NC} (PID: ${pid})"
        echo -e "  URL:      ${CYAN}http://127.0.0.1:${port}${NC}"
        echo -e "  Log file: ${BLUE}${LOG_FILE}${NC}"
        echo -e "  $(llm_summary_info "${info}")"
        echo -e "\n${BOLD}Recent Log Output (last 8 lines):${NC}"
        tail -n 8 "${LOG_FILE}" 2>/dev/null || echo "No logs yet."
    else
        echo -e "${RED}○ Chronicle dashboard is STOPPED${NC}"
        rm -f "${PID_FILE}" "${PORT_FILE}" "${LLM_FILE}"
    fi

    if mission_running; then
        local mpid mport
        mpid=$(mission_pid)
        mport="${MISSION_DEFAULT_PORT}"
        [ -f "${MISSION_PORT_FILE}" ] && mport=$(cat "${MISSION_PORT_FILE}")
        echo -e "\n${GREEN}● Zine mission control is RUNNING${NC} (PID: ${mpid})"
        echo -e "  URL:      ${CYAN}http://127.0.0.1:${mport}${NC}"
        echo -e "  Log file: ${BLUE}${MISSION_LOG_FILE}${NC}"
    else
        echo -e "\n${RED}○ Zine mission control is STOPPED${NC} (start it with: ./chronicle.sh mission)"
        rm -f "${MISSION_PID_FILE}" "${MISSION_PORT_FILE}"
    fi
}

serve_foreground() {
    local source="${1:-$DEFAULT_SOURCE}"
    local port="${2:-$DEFAULT_PORT}"

    echo -e "${BOLD}${CYAN}Launching Chronicle dashboard in foreground...${NC}"
    echo -e "  $(llm_summary)"
    echo -e "Access at: ${CYAN}http://127.0.0.1:${port}${NC} (Press Ctrl+C to stop)\n"

    exec "${CHRONICLE_BIN}" serve \
        --source "${source}" \
        --port "${port}" \
        --host "127.0.0.1" \
        "${LLM_ARGS[@]}"
}

show_logs() {
    if [ ! -f "${LOG_FILE}" ]; then
        echo "No log file found at ${LOG_FILE}."
        exit 0
    fi
    echo -e "${BOLD}Following logs from ${BLUE}${LOG_FILE}${NC} (Press Ctrl+C to exit):"
    tail -f "${LOG_FILE}"
}

# ------------------------------------------------------------------------------
# Zine mission control (final walkthrough dashboard from run_dashboard.py)
# ------------------------------------------------------------------------------

mission_pid() {
    if [ -f "${MISSION_PID_FILE}" ]; then
        cat "${MISSION_PID_FILE}"
    fi
}

mission_running() {
    if [ -f "${MISSION_PID_FILE}" ]; then
        local pid
        pid=$(cat "${MISSION_PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

mission_llm_env() {
    # Map the chronicle.sh LLM flags onto run_dashboard.py env vars.
    local info
    info=$(effective_llm)
    local backend="${info%%|*}"
    local model="${info##*|}"
    case "${backend}" in
        ollama)     export CHRONICLE_LLM_BACKEND="ollama" ;;
        template)   export CHRONICLE_LLM_BACKEND="template" ;;
        transformers) export CHRONICLE_LLM_BACKEND="template" ;;
        none)       export CHRONICLE_LLM_BACKEND="none" ;;
        *)          export CHRONICLE_LLM_BACKEND="auto" ;;
    esac
    export CHRONICLE_LLM_MODEL="${model}"
    export CHRONICLE_OLLAMA_URL="${OLLAMA_URL}"
}

start_mission() {
    local source="${1:-$DEFAULT_SOURCE}"
    local port="${2:-$MISSION_DEFAULT_PORT}"

    if mission_running; then
        echo -e "${YELLOW}Mission control already running (PID: $(mission_pid)).${NC}"
        echo -e "Open the zine dashboard at: ${CYAN}http://127.0.0.1:${port}${NC}"
        return 0
    fi

    local port_pid
    port_pid=$(find_port_pid "${port}")
    if [ -n "${port_pid}" ]; then
        echo -e "${RED}Port ${port} is currently in use by process ${port_pid}.${NC}"
        echo "Stop it first or pass a different port: ./chronicle.sh mission ${port}"
        return 1
    fi

    mission_llm_env

    echo -e "${BOLD}${CYAN}Starting Chronicle zine mission control...${NC}"
    echo -e "  Source:   ${GREEN}${source}${NC}"
    echo -e "  Port:     ${GREEN}${port}${NC}"
    echo -e "  $(llm_summary)"

    # The mission control replays the bundled demo episode. Any other source
    # name runs in live mode and waits for POST /api/ingest events.
    local dash_source="demo"
    if [ "${source}" != "synthetic" ]; then
        dash_source="live"
        echo -e "  ${YELLOW}note: ${source} has no bundled demo replay — starting in live mode ${NC}"
        echo -e "  ${YELLOW}(feed it with POST /api/ingest, CHRONICLE_EVENTS_JSONL or CHRONICLE_DB)${NC}"
    fi

    nohup setsid "${PYTHON_BIN}" run_dashboard.py \
        --source "${dash_source}" \
        --port "${port}" \
        --host "127.0.0.1" \
        >> "${MISSION_LOG_FILE}" 2>&1 < /dev/null &

    local pid=$!
    disown "${pid}" 2>/dev/null || true
    echo "${pid}" > "${MISSION_PID_FILE}"
    echo "${port}" > "${MISSION_PORT_FILE}"

    # Wait up to 8 seconds for NiceGUI to bind
    local timeout=16
    local ready=false
    while [ "${timeout}" -gt 0 ]; do
        if grep -q "NiceGUI ready" "${MISSION_LOG_FILE}" 2>/dev/null; then
            ready=true
            break
        fi
        sleep 0.5
        timeout=$((timeout - 1))
    done

    if kill -0 "${pid}" 2>/dev/null && curl -sf --max-time 2 "http://127.0.0.1:${port}/api/health" >/dev/null 2>&1; then
        ready=true
    fi

    if [ "${ready}" = true ]; then
        echo -e "${GREEN}✓ Zine mission control started!${NC} (PID: ${pid})"
        echo -e "\n  ${BOLD}URL:${NC} ${CYAN}http://127.0.0.1:${port}${NC}\n"
        echo -e "\nUseful commands:"
        echo -e "  ${YELLOW}./chronicle.sh mission-stop${NC}   - Stop the zine dashboard"
        echo -e "  ${YELLOW}./chronicle.sh mission-logs${NC}   - View live logs"
        echo -e "  ${YELLOW}./chronicle.sh start${NC}          - Run the full pipeline UI on :8080"
        return 0
    fi

    echo -e "${RED}✗ Mission control failed to start. Check logs:${NC}"
    tail -n 15 "${MISSION_LOG_FILE}"
    rm -f "${MISSION_PID_FILE}" "${MISSION_PORT_FILE}"
    return 1
}

stop_mission() {
    if [ -f "${MISSION_PID_FILE}" ]; then
        local pid
        pid=$(cat "${MISSION_PID_FILE}")
        echo -e "${YELLOW}Stopping zine mission control (PID: ${pid})...${NC}"
        if kill -0 "${pid}" 2>/dev/null; then
            kill -15 "${pid}" 2>/dev/null || true
            for _ in {1..10}; do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 0.3
            done
            kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${MISSION_PID_FILE}" "${MISSION_PORT_FILE}"
        echo -e "${GREEN}✓ Stopped.${NC}"
    else
        # Fall back to reporting whatever runs on the mission port.
        local port="${1:-$MISSION_DEFAULT_PORT}"
        local port_pid
        port_pid=$(find_port_pid "${port}")
        if [ -n "${port_pid}" ]; then
            echo -e "${YELLOW}No mission pid file, but PID ${port_pid} is still bound to :${port}.${NC}"
            echo -e "Stop it manually with:  kill ${port_pid}"
        else
            echo "Mission control was not running."
            rm -f "${MISSION_PID_FILE}" "${MISSION_PORT_FILE}"
        fi
    fi
}

show_mission_logs() {
    if [ ! -f "${MISSION_LOG_FILE}" ]; then
        echo "No log file found at ${MISSION_LOG_FILE}."
        exit 0
    fi
    echo -e "${BOLD}Following logs from ${BLUE}${MISSION_LOG_FILE}${NC} (Press Ctrl+C to exit):"
    tail -f "${MISSION_LOG_FILE}"
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
    echo -e "  ${GREEN}mission${NC} [source] [port] Start the zine mission control (run_dashboard.py, port ${CYAN}8081${NC})"
    echo -e "                          ${CYAN}synthetic${NC} replays the bundled demo; other sources wait for live ingest"
    echo -e "  ${GREEN}mission-stop${NC}            Stop the zine mission control"
    echo -e "  ${GREEN}mission-logs${NC}            Tail zine mission control logs (-f)"
    echo -e "  ${GREEN}serve${NC} [source] [port]   Run the web dashboard in foreground"
    echo -e "  ${GREEN}test${NC}                    Run full test & verification suite"
    echo -e "  ${GREEN}preflight${NC} [options]     Check videos, SQLite, backends, LLM, tests"
    echo -e "                          and a 30-frame run before the demo"
    echo -e "  ${GREEN}sources${NC}                 List all registered video sources"
    echo -e "  ${GREEN}help${NC}                    Show this help message"
    echo
    echo -e "${BOLD}LLM options${NC} (start/restart/serve):"
    echo -e "  ${GREEN}--no-llm${NC}               Deterministic graph answers (no Ollama)"
    echo -e "  ${GREEN}--llm${NC} BACKEND          ${CYAN}auto${NC}|${CYAN}ollama${NC}|${CYAN}transformers${NC}|${CYAN}template${NC} (default: ${LLM_BACKEND})"
    echo -e "  ${GREEN}--llm-model${NC} NAME       Ollama model (default: ${LLM_MODEL})"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  ./chronicle.sh start                    # Start with synthetic fixture on port 8080"
    echo -e "  ./chronicle.sh start tabletop           # Start with HoloLens tabletop on port 8080"
    echo -e "  ./chronicle.sh start cook 8085          # Start with Aria cook clip on port 8085"
    echo -e "  ./chronicle.sh start --llm-model qwen2.5:1.5b  # Faster local model"
    echo -e "  ./chronicle.sh start --no-llm           # Deterministic answers only"
    echo -e "  ./chronicle.sh mission                  # Zine mission control (demo replay, :8081)"
    echo -e "  ./chronicle.sh mission --no-llm 8090    # Same, deterministic, custom port"
    echo -e "  ./chronicle.sh mission-stop             # Stop the zine dashboard"
    echo -e "  ./chronicle.sh stop                     # Stop running server"
    echo -e "  ./chronicle.sh status                   # Check if running"
    echo -e "  ./chronicle.sh test                     # Run all 250+ tests & benchmarks"
    echo -e "  ./chronicle.sh preflight                # Check demo readiness"
    echo -e "  ./chronicle.sh preflight --warm-llm     # Load Qwen before the demo"
}

# ------------------------------------------------------------------------------
# Command Dispatcher
# ------------------------------------------------------------------------------
ACTION="${1:-launch}"
if [[ "${ACTION}" == -* ]]; then
    # Option-only invocation, e.g. `./chronicle.sh --no-llm`, means launch.
    ACTION="launch"
else
    shift || true
fi

case "${ACTION}" in
    launch|start)
        parse_llm_flags "$@"
        start_server "${PARSE_POSITIONAL[0]:-$DEFAULT_SOURCE}" "${PARSE_POSITIONAL[1]:-$DEFAULT_PORT}"
        ;;
    mission)
        parse_llm_flags "$@"
        m_port="${MISSION_DEFAULT_PORT}"
        m_source="${DEFAULT_SOURCE}"
        if [ -n "${PARSE_POSITIONAL[0]:-}" ]; then
            if [[ "${PARSE_POSITIONAL[0]}" =~ ^[0-9]+$ ]]; then
                m_port="${PARSE_POSITIONAL[0]}"
            else
                m_source="${PARSE_POSITIONAL[0]}"
            fi
        fi
        if [ -n "${PARSE_POSITIONAL[1]:-}" ]; then
            m_port="${PARSE_POSITIONAL[1]}"
        fi
        start_mission "${m_source}" "${m_port}"
        ;;
    mission-stop|mission_off)
        stop_mission "${1:-$MISSION_DEFAULT_PORT}"
        ;;
    mission-logs|mission-log)
        show_mission_logs
        ;;
    stop)
        stop_server "${1:-$DEFAULT_PORT}"
        ;;
    restart)
        parse_llm_flags "$@"
        restart_source="${PARSE_POSITIONAL[0]:-$DEFAULT_SOURCE}"
        restart_port="${PARSE_POSITIONAL[1]:-$DEFAULT_PORT}"
        stop_server "${restart_port}"
        start_server "${restart_source}" "${restart_port}"
        ;;
    status)
        server_status
        ;;
    logs|log)
        show_logs
        ;;
    serve|run)
        parse_llm_flags "$@"
        serve_foreground "${PARSE_POSITIONAL[0]:-$DEFAULT_SOURCE}" "${PARSE_POSITIONAL[1]:-$DEFAULT_PORT}"
        ;;
    test|tests)
        run_tests
        ;;
    preflight|doctor)
        exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/demo_preflight.py" "$@"
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
