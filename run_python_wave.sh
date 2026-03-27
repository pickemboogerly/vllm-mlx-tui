#!/usr/bin/env bash
set -e

# Ensure we're relative to the script location
cd "$(dirname "$0")/python-wave"

echo "🐍 Setting up Python environment for vLLM-wave..."

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install the package in editable mode along with its dependencies
echo "Installing dependencies..."
pip install -e . --quiet

echo "✨ Launching vLLM-wave (Python / Textual)..."
echo "--------------------------------------------------------"
vllm-wave
