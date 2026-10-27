// Package engine wires every component into a forward pass and a
// generation loop. One Engine = one loaded model. The per-token path:
//
//	embed + position → for each layer:
//	    x += Attention(RMSNorm(x))          — dense, always in RAM
//	    x += MoE(RMSNorm(x))                — routed experts, tiered store
//	→ logits = RMSNorm(x) · embedᵀ          — weight-tied LM head
//	→ sample next token
package engine

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"sort"

	"algo-engine/internal/attention"
	"algo-engine/internal/expertstore"
	"algo-engine/internal/heatmap"
	"algo-engine/internal/kvcache"
	"algo-engine/internal/pipeline"
	"algo-engine/internal/quant"
	"algo-engine/internal/router"
	"algo-engine/internal/safetensors"
	"algo-engine/internal/tokenizer"
)

// layerWeights bundles one transformer layer's dense pieces.
type layerWeights struct {
	attn     *attention.Layer
	gate     *router.Gate
	normAttn []float32
	normFFN  []float32
}

// Engine is a fully loaded model plus its tiered expert store.
type Engine struct {
	Cfg       ModelConfig
	Tok       *tokenizer.Tokenizer
	Heat      *heatmap.HeatMap
	Store     *expertstore.Store
	embed     []float32 // [vocab × dModel], also the tied LM head
	layers    []layerWeights
	normFinal []float32
	moe       *pipeline.MoE
	kernel    quant.Kernel
	expertsF  *safetensors.File // kept open for streaming; closed by Close
}

// KernelTier reports which quant.Kernel build is actually running —
// "pure-go", "cpu-accel", or "cuda" — for GET /stats (internal/api) and
// the performance_optimizer dashboard's GPU panel.
func (e *Engine) KernelTier() string { return e.kernel.Name() }

// Options tune the runtime (not the model).
type Options struct {
	CacheCapacity      int   // RAM tier size, in experts
	MaxConcurrentReads int   // disk streaming parallelism
	PinHottest         int   // pin the N hottest experts from usage.json
	Seed               int64 // sampling seed (0 = fixed default)
}

// Load opens a model directory and builds a ready Engine.
func Load(modelDir string, opt Options) (*Engine, error) {
	cfg, err := LoadConfig(modelDir)
	if err != nil {
		return nil, err
	}

	dense, err := safetensors.Open(filepath.Join(modelDir, "dense.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("open dense weights: %w", err)
	}
	defer dense.Close() // fully read into RAM below; file not needed after

	expertsF, err := safetensors.Open(filepath.Join(modelDir, "experts.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("open expert weights: %w", err)
	}

	e := &Engine{Cfg: cfg, kernel: quant.Active(), expertsF: expertsF,
		Heat: heatmap.New()}

	// Heat map: load persisted usage so pins reflect real history.
	usagePath := filepath.Join(modelDir, "usage.json")
	if err := e.Heat.Load(usagePath); err != nil {
		return nil, err
	}

	// Dense weights → RAM.
	if e.embed, err = dense.ReadF32("embed"); err != nil {
		return nil, err
	}
	for l := 0; l < cfg.NLayers; l++ {
		p := fmt.Sprintf("layer.%d.", l)
		read := func(name string) []float32 {
			if err != nil {
				return nil
			}
			var t []float32
			t, err = dense.ReadF32(p + name)
			return t
		}
		lw := layerWeights{
			attn: attention.NewLayer(read("wq"), read("wk"), read("wv"),
				read("wo"), cfg.DModel),
			normAttn: read("norm_attn"),
			normFFN:  read("norm_ffn"),
		}
		lw.gate = router.NewGate(read("gate"), cfg.NExperts, cfg.DModel, cfg.TopK)
		if err != nil {
			return nil, fmt.Errorf("layer %d dense weights: %w", l, err)
		}
		e.layers = append(e.layers, lw)
	}
	if e.normFinal, err = dense.ReadF32("norm_final"); err != nil {
		return nil, err
	}

	// Tiered expert store over the still-open experts file.
	layout := expertstore.Layout{DModel: cfg.DModel, DFF: cfg.DFF}
	e.Store = expertstore.New(expertsF, layout, e.Heat, expertstore.Config{
		Capacity:           opt.CacheCapacity,
		MaxConcurrentReads: opt.MaxConcurrentReads,
	})
	e.moe = &pipeline.MoE{Store: e.Store, Kernel: e.kernel, Layout: layout}

	// Learned pins: warm the hottest experts from history.
	if opt.PinHottest > 0 {
		if err := e.Store.Pin(context.Background(),
			e.Heat.Hottest(opt.PinHottest)); err != nil {
			return nil, err
		}
	}

	e.Tok = tokenizer.NewByteLevel(nil) // toy model: pure byte-level
	return e, nil
}

// Close releases the expert file and persists the heat map.
func (e *Engine) Close(modelDir string) error {
	if err := e.Heat.Save(filepath.Join(modelDir, "usage.json")); err != nil {
		return err
	}
	return e.expertsF.Close()
}

// ── Forward pass ─────────────────────────────────────────────────────────

// rmsNorm normalizes x by its root-mean-square, scaled by learned gains g:
// the modern, simpler cousin of LayerNorm (no mean subtraction, no bias).
func rmsNorm(x, g []float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(ss/float64(len(x))+1e-6)
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v)*inv) * g[i]
	}
	return out
}

// positional returns a sinusoidal position encoding — pos is visible to
// the model through addition at the embedding. (Real models use RoPE
// inside attention; sinusoidal keeps the toy readable.)
func positional(pos, d int) []float32 {
	out := make([]float32, d)
	for i := 0; i < d; i += 2 {
		freq := math.Pow(10000, -float64(i)/float64(d))
		angle := float64(pos) * freq
		out[i] = float32(math.Sin(angle))
		if i+1 < d {
			out[i+1] = float32(math.Cos(angle))
		}
	}
	return out
}

// ForwardToken runs one token through all layers, extending the KV cache,
// and returns vocabulary logits for the next-token prediction.
func (e *Engine) ForwardToken(ctx context.Context, tokID, pos int,
	cache *kvcache.Cache) ([]float32, error) {

	d := e.Cfg.DModel
	x := make([]float32, d)
	copy(x, e.embed[tokID*d:(tokID+1)*d])
	for i, p := range positional(pos, d) {
		x[i] += p
	}

	for li, lw := range e.layers {
		// attention block with residual connection
		h := rmsNorm(x, lw.normAttn)
		a := lw.attn.Forward(h, cache, li)
		for i := range x {
			x[i] += a[i]
		}

		// MoE block with residual connection
		h = rmsNorm(x, lw.normFFN)
		choices := lw.gate.Route(h)
		y, err := e.moe.Forward(ctx, li, h, choices)
		if err != nil {
			return nil, err
		}
		for i := range x {
			x[i] += y[i]
		}
	}

	// weight-tied LM head: logits = normed(x) · embedᵀ
	h := rmsNorm(x, e.normFinal)
	return e.kernel.MatVec(e.embed, e.Cfg.VocabSize, d, h), nil
}

// ── Sampling ─────────────────────────────────────────────────────────────

// Sampler picks the next token from logits.
type Sampler struct {
	Temperature float64 // 0 → greedy argmax
	TopK        int     // 0 → full distribution
	rng         *rand.Rand
}

func NewSampler(temperature float64, topK int, seed int64) *Sampler {
	if seed == 0 {
		seed = 1
	}
	return &Sampler{Temperature: temperature, TopK: topK,
		rng: rand.New(rand.NewSource(seed))}
}

// Next samples one token ID.
func (s *Sampler) Next(logits []float32) int {
	if s.Temperature <= 0 {
		best, bestV := 0, float32(math.Inf(-1))
		for i, v := range logits {
			if v > bestV {
				best, bestV = i, v
			}
		}
		return best
	}

	type cand struct {
		id int
		l  float64
	}
	cands := make([]cand, len(logits))
	for i, v := range logits {
		cands[i] = cand{i, float64(v) / s.Temperature}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].l > cands[b].l })
	if s.TopK > 0 && s.TopK < len(cands) {
		cands = cands[:s.TopK]
	}

	maxL := cands[0].l
	var sum float64
	probs := make([]float64, len(cands))
	for i, c := range cands {
		probs[i] = math.Exp(c.l - maxL)
		sum += probs[i]
	}
	r := s.rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return cands[i].id
		}
	}
	return cands[len(cands)-1].id
}

// ── Generation ───────────────────────────────────────────────────────────

// Generate prefills the prompt, then decodes up to maxNew tokens, calling
// onToken for each (nil is fine). Returns the generated IDs.
func (e *Engine) Generate(ctx context.Context, prompt string, maxNew int,
	sampler *Sampler, cache *kvcache.Cache,
	onToken func(id int, text string)) ([]int, error) {

	ids := append([]int{tokenizer.BOS}, e.Tok.Encode(prompt)...)

	// Prefill: run every prompt token; only the last logits matter, but
	// every position must populate the KV cache.
	var logits []float32
	pos := cache.Len()
	for _, id := range ids {
		if pos >= e.Cfg.MaxSeq {
			return nil, fmt.Errorf("context full (%d)", e.Cfg.MaxSeq)
		}
		var err error
		logits, err = e.ForwardToken(ctx, id, pos, cache)
		if err != nil {
			return nil, err
		}
		pos++
	}

	// Decode loop.
	var out []int
	for n := 0; n < maxNew; n++ {
		if err := ctx.Err(); err != nil {
			return out, err
		}
		next := sampler.Next(logits)
		if next == tokenizer.EOS {
			break
		}
		out = append(out, next)
		if onToken != nil {
			onToken(next, e.Tok.Decode([]int{next}))
		}
		if pos >= e.Cfg.MaxSeq {
			break
		}
		var err error
		logits, err = e.ForwardToken(ctx, next, pos, cache)
		if err != nil {
			return out, err
		}
		pos++
	}
	return out, nil
}
