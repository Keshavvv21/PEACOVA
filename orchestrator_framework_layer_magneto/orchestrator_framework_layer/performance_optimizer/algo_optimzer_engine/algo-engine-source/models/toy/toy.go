// Package toy is the first ModelEngine family: the manual's toy MoE,
// implemented as a thin adapter over internal/engine. It exists to prove
// the seam — the same interface the OLMoE adapter fills in — and to keep
// the toy runnable forever as the pipeline's smoke test, exactly the role
// 's smallest family plays there.
package toy

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"algo-engine/internal/engine"
	"algo-engine/internal/kvcache"
	"algo-engine/models"
)

func init() { models.Register(Family{}) }

// Family implements models.ModelEngine for the toy MoE.
type Family struct{}

func (Family) Family() string { return "toy" }

// Detect: a toy directory has our config.json naming the toy model.
func (Family) Detect(modelDir string) bool {
	cfg, err := engine.LoadConfig(modelDir)
	return err == nil && cfg.Name == "toy-moe"
}

func (Family) Describe(modelDir string) (string, error) {
	cfg, err := engine.LoadConfig(modelDir)
	if err != nil {
		return "", err
	}
	var disk int64
	for _, f := range []string{"dense.safetensors", "experts.safetensors"} {
		if st, err := os.Stat(filepath.Join(modelDir, f)); err == nil {
			disk += st.Size()
		}
	}
	return fmt.Sprintf("%s: %d layers × %d experts (top-%d), d=%d, int8, %.1f MB on disk",
		cfg.Name, cfg.NLayers, cfg.NExperts, cfg.TopK, cfg.DModel,
		float64(disk)/1e6), nil
}

func (Family) Template() models.ChatTemplate { return template{} }

// template is the toy's honest scaffolding: role-tagged lines and an
// assistant cue. Real families ship real templates — see models/olmoe.
type template struct{}

func (template) Render(msgs []models.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&b, "%s: %s\n", m.Role, m.Content)
	}
	b.WriteString("assistant:")
	return b.String()
}

func (template) Stop() []string { return nil }

func (Family) Generate(modelDir, session, prompt string, maxNew int,
	temperature float64, onToken func(text string)) (string, error) {

	eng, err := engine.Load(modelDir, engine.Options{
		CacheCapacity: 32, MaxConcurrentReads: 4, PinHottest: 8})
	if err != nil {
		return "", err
	}
	defer eng.Close(modelDir)

	cache := kvcache.New(session, eng.Cfg.NLayers)
	if session != "" {
		if cache, err = kvcache.Load(modelDir, session, eng.Cfg.NLayers); err != nil {
			return "", err
		}
	}

	var out strings.Builder
	cb := func(_ int, text string) {
		out.WriteString(text)
		if onToken != nil {
			onToken(text)
		}
	}
	_, err = eng.Generate(context.Background(), prompt, maxNew,
		engine.NewSampler(temperature, 40, time.Now().UnixNano()), cache, cb)
	if err != nil {
		return out.String(), err
	}
	if session != "" {
		if err := cache.Save(modelDir); err != nil {
			return out.String(), err
		}
	}
	return out.String(), nil
}
