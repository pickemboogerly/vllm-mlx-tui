#!/usr/bin/env bash
set -e

# Ensure we're relative to the script location
cd "$(dirname "$0")/go-wave"

echo "🐹 Setting up Go environment for vLLM-wave..."

# Tidy dependencies and build the binary
echo "Compiling..."
go mod tidy
go build -o vllm-wave-bin .

echo "✨ Launching vLLM-wave (Go / Bubble Tea)..."
echo "--------------------------------------------------------"
./vllm-wave-bin
