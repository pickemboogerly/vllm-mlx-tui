package cache

import (
	"os"
	"path/filepath"
	"strings"
)

// ListCachedModels scans the HuggingFace cache directory for models.
func ListCachedModels() []string {
	var models []string
	home, err := os.UserHomeDir()
	if err != nil {
		return defaultModels()
	}

	cacheDir := filepath.Join(home, ".cache", "huggingface", "hub")
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return defaultModels()
	}

	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			modelID := strings.TrimPrefix(entry.Name(), "models--")
			modelID = strings.ReplaceAll(modelID, "--", "/")
			models = append(models, modelID)
		}
	}

	if len(models) == 0 {
		return defaultModels()
	}

	return models
}

func defaultModels() []string {
	return []string{
		"mlx-community/Meta-Llama-3-8B-Instruct-4bit",
		"mlx-community/Mistral-7B-Instruct-v0.2-4bit",
		"mlx-community/Phi-3-mini-4k-instruct-4bit",
	}
}
