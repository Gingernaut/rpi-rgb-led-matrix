#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'Usage: ./run_server.sh [OPTIONS]

Description:
    Start the Pixel Display API server (FastAPI + uvicorn).
    Runs with sudo for GPIO access and realtime thread priority.

Options:
    -h, --help          Show this help message and exit
    --host <host>       Bind address (default: "0.0.0.0")
    --port <port>       Bind port (default: 8000)
    --reload            Enable auto-reload for development
    --no-sudo           Run without sudo (colors will be degraded)

Environment Variables:
    TRACE=1             Enable debug tracing

Examples:
    ./run_server.sh
    ./run_server.sh --port 3000
    ./run_server.sh --no-sudo --reload
'
    exit 0
fi

cd "$(dirname "$0")"

function main() {
    HOST="0.0.0.0"
    PORT="8000"
    RELOAD=""
    USE_SUDO=1

    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --reload)
                RELOAD="--reload"
                shift
                ;;
            --no-sudo)
                USE_SUDO=0
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    if [[ "${USE_SUDO}" -eq 1 ]]; then
        exec sudo --preserve-env uv run uvicorn server.app:app --host "${HOST}" --port "${PORT}" ${RELOAD}
    else
        exec uv run uvicorn server.app:app --host "${HOST}" --port "${PORT}" ${RELOAD}
    fi
}

main "$@"
