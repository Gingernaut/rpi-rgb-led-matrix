#!/usr/bin/env bash
# Setup script for rpi-rgb-led-matrix on a Raspberry Pi.
# Installs pyenv, builds the C library and Python bindings,
# creates a venv with project dependencies.

set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'Usage: ./setup.sh

Description:
    Set up the rpi-rgb-led-matrix project from scratch.
    Installs pyenv (if missing), the correct Python version,
    builds the C library and Python bindings, creates a venv,
    and installs project dependencies.

    This script is idempotent and can be re-run safely.

Environment Variables:
    TRACE=1    Enable debug tracing
'
    exit 0
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

PYTHON_DIR="${REPO_DIR}/bindings/python"

####################### pyenv #######################

install_pyenv() {
    echo "===== pyenv ====="

    if command -v pyenv &>/dev/null; then
        echo "  pyenv already installed."
    else
        echo "  Installing pyenv..."
        curl https://pyenv.run | bash
    fi

    # Ensure pyenv is available in this shell
    export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
    export PATH="${PYENV_ROOT}/bin:${PATH}"
    eval "$(pyenv init -)"

    echo ""
}

####################### Python version #######################

setup_python() {
    echo "===== Python ====="

    local py_version
    py_version=$(<.python-version)
    py_version=${py_version//[[:space:]]/}

    local installed_version
    installed_version=$(pyenv versions --bare | grep "^${py_version}" | tail -1)

    if [[ -z "$installed_version" ]]; then
        echo "  Installing Python ${py_version} via pyenv..."
        pyenv install "${py_version}"
        installed_version=$(pyenv versions --bare | grep "^${py_version}" | tail -1)
    else
        echo "  Python ${installed_version} already installed."
    fi

    pyenv local "${installed_version}"
    echo "  Using Python ${installed_version}"
    echo ""
}

####################### Virtual environment #######################

setup_venv() {
    echo "===== Virtual Environment ====="

    cd "${PYTHON_DIR}"

    # Use nvenv if available (from dotfiles), otherwise create venv manually
    if type nvenv &>/dev/null; then
        echo "  Creating venv via nvenv..."
        nvenv
    else
        echo "  Creating venv..."
        python3 -m venv .venv --upgrade-deps
        source .venv/bin/activate
    fi

    echo ""
}

####################### Build #######################

build_project() {
    echo "===== Build ====="

    cd "${REPO_DIR}"

    # Build the C library and Python bindings (run from repo root per upstream docs)
    echo "  Building C library and Python bindings..."
    make build-python PYTHON="$(which python3)"
    sudo make install-python PYTHON="$(which python3)"

    echo "  Build complete."
    echo ""
}

####################### Project dependencies #######################

install_deps() {
    echo "===== Project Dependencies ====="

    cd "${PYTHON_DIR}"

    if command -v poetry &>/dev/null; then
        echo "  Installing dependencies via poetry..."
        poetry install
    else
        echo "  Poetry not found, installing via pip..."
        pip install -r <(sed -n '/^\[tool.poetry.dependencies\]/,/^\[/{/^[a-z]/s/ = .*//p}' pyproject.toml)
    fi

    echo "  Dependencies installed."
    echo ""
}

####################### Verification #######################

verify() {
    echo "===== Verification ====="

    echo "  Python:  $(python3 --version)"
    echo "  pyenv:   $(pyenv --version)"
    echo "  venv:    ${VIRTUAL_ENV:-not active}"
    echo ""

    echo "  To run the webserver:"
    echo "    cd ${PYTHON_DIR}"
    echo "    source .venv/bin/activate"
    echo "    sudo python main.py"
    echo ""
}

####################### Main #######################

main() {
    echo ""
    echo "=========================================="
    echo "  rpi-rgb-led-matrix Setup"
    echo "=========================================="
    echo ""

    install_pyenv
    setup_python
    setup_venv
    build_project
    install_deps
    verify
}

main "$@"
