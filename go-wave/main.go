package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("========================================================")
	fmt.Println("⚠️ WARNING: The Go implementation of vLLM-wave (go-wave)")
	fmt.Println("has been officially DEPRECATED as a failed experiment.")
	fmt.Println("Please use the Python version instead.")
	fmt.Println()
	fmt.Println("To run the canonical version: ./run_python_wave.sh")
	fmt.Println("========================================================")
	os.Exit(1)
}
