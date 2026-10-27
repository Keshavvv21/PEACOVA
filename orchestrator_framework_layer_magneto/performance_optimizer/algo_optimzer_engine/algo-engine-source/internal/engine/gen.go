package engine

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"algo-engine/internal/quant"
	"algo-engine/internal/safetensors"
)

// GenerateToyModel writes a complete, loadable model directory with
// deterministic random weights:
//
//	<dir>/config.json           — the ModelConfig
//	<dir>/dense.safetensors     — F32 dense weights (embeddings, attention,
//	                              router gates, norms) — the RAM-resident set
//	<dir>/experts.safetensors   — I8-quantized experts + F32 scales — the
//	                              disk-streamed set
//
// The split mirrors  exactly: dense weights are small and always
// resident; the expert long tail is quantized and streamed. Same seed →
// byte-identical files → reproducible generations.
func GenerateToyModel(dir string, seed int64) error {
	cfg := ToyConfig()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := SaveConfig(dir, cfg); err != nil {
		return err
	}
	rng := rand.New(rand.NewSource(seed))

	// Xavier-style scaling keeps activations from exploding through
	// untrained layers: stddev = 1/sqrt(fan_in).
	randMat := func(rows, cols int) []float32 {
		std := 1 / math.Sqrt(float64(cols))
		m := make([]float32, rows*cols)
		for i := range m {
			m[i] = float32(rng.NormFloat64() * std)
		}
		return m
	}
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}

	// ── dense.safetensors ────────────────────────────────────────────
	dense := safetensors.NewWriter()
	d, ff := cfg.DModel, cfg.DFF
	_ = ff
	dense.AddF32("embed", []int64{int64(cfg.VocabSize), int64(d)},
		randMat(cfg.VocabSize, d))
	for l := 0; l < cfg.NLayers; l++ {
		p := fmt.Sprintf("layer.%d.", l)
		dense.AddF32(p+"wq", []int64{int64(d), int64(d)}, randMat(d, d))
		dense.AddF32(p+"wk", []int64{int64(d), int64(d)}, randMat(d, d))
		dense.AddF32(p+"wv", []int64{int64(d), int64(d)}, randMat(d, d))
		dense.AddF32(p+"wo", []int64{int64(d), int64(d)}, randMat(d, d))
		dense.AddF32(p+"gate", []int64{int64(cfg.NExperts), int64(d)},
			randMat(cfg.NExperts, d))
		dense.AddF32(p+"norm_attn", []int64{int64(d)}, ones(d))
		dense.AddF32(p+"norm_ffn", []int64{int64(d)}, ones(d))
	}
	dense.AddF32("norm_final", []int64{int64(d)}, ones(d))
	if err := dense.WriteFile(filepath.Join(dir, "dense.safetensors")); err != nil {
		return fmt.Errorf("write dense: %w", err)
	}

	// ── experts.safetensors ──────────────────────────────────────────
	experts := safetensors.NewWriter()
	for l := 0; l < cfg.NLayers; l++ {
		for e := 0; e < cfg.NExperts; e++ {
			w1 := randMat(cfg.DFF, cfg.DModel)
			w2 := randMat(cfg.DModel, cfg.DFF)
			q1, s1 := quant.Quantize8(w1)
			q2, s2 := quant.Quantize8(w2)
			raw := append(quant.BytesFromInt8(q1), quant.BytesFromInt8(q2)...)
			name := fmt.Sprintf("expert.%d.%d", l, e)
			experts.AddI8(name, []int64{int64(len(raw))}, raw)
			experts.AddF32(name+".scales", []int64{2}, []float32{s1, s2})
		}
	}
	if err := experts.WriteFile(filepath.Join(dir, "experts.safetensors")); err != nil {
		return fmt.Errorf("write experts: %w", err)
	}
	return nil
}
