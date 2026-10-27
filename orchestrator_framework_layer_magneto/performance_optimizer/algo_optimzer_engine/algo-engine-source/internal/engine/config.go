package engine

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// ModelConfig mirrors config.json in the model directory. The toy defaults
// are small enough that a full forward pass takes microseconds, while every
// mechanism (routing, streaming, caching) stays fully real.
type ModelConfig struct {
	Name      string `json:"name"`
	DModel    int    `json:"d_model"`
	DFF       int    `json:"d_ff"`
	NLayers   int    `json:"n_layers"`
	NExperts  int    `json:"n_experts"`
	TopK      int    `json:"top_k"`
	VocabSize int    `json:"vocab_size"`
	MaxSeq    int    `json:"max_seq"`
}

// ToyConfig is the reference toy model this manual builds against.
func ToyConfig() ModelConfig {
	return ModelConfig{
		Name: "toy-moe", DModel: 64, DFF: 128,
		NLayers: 4, NExperts: 16, TopK: 2,
		VocabSize: 260, MaxSeq: 512,
	}
}

// LoadConfig reads config.json from a model directory.
func LoadConfig(modelDir string) (ModelConfig, error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return ModelConfig{}, fmt.Errorf("read config.json: %w", err)
	}
	var c ModelConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return ModelConfig{}, fmt.Errorf("parse config.json: %w", err)
	}
	if c.DModel <= 0 || c.NLayers <= 0 || c.NExperts < c.TopK || c.TopK <= 0 {
		return ModelConfig{}, fmt.Errorf("config.json fails sanity checks: %+v", c)
	}
	return c, nil
}

// SaveConfig writes config.json (used by the toy weight generator).
func SaveConfig(modelDir string, c ModelConfig) error {
	raw, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(modelDir, "config.json"), raw, 0o644)
}
