#!/bin/bash
set -e

# Change to the root of the project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Ensuring latest version of vllm-mlx-tui..."

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install the package in editable mode to ensure the latest source is used
# Also ensures all dependencies are up to date from pyproject.toml
pip install -q -e .

echo "✅ Ready! Launching TUI..."
vllm-mlx-tui
