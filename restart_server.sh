#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

sudo systemctl restart led-matrix.service
echo "led-matrix.service restarted"
