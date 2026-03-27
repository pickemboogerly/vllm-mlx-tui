package cache

import (
	"os"
	"path/filepath"
	"strings"
)

// ListCachedModels scans the HuggingFace cache directory for models.
// Returns an empty slice if no models are found; the caller is responsible
// for handling the empty case gracefully (L6: no fake fallback models).
func ListCachedModels() []string {
	var models []string
	home, err := os.UserHomeDir()
	if err != nil {
		return models
	}

	cacheDir := filepath.Join(home, ".cache", "huggingface", "hub")
	entries, err := os.ReadDir(cacheDir)
	if err != nil {
		return models
	}

	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			modelID := strings.TrimPrefix(entry.Name(), "models--")
			modelID = strings.ReplaceAll(modelID, "--", "/")
			models = append(models, modelID)
		}
	}

	return models
}
