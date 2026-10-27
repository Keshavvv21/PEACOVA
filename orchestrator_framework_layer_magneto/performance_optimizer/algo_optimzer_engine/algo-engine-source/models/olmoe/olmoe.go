// Package olmoe is the adapter for the first REAL SLM target: OLMoE
// (AllenAI's 7B-total / 1B-active open MoE — 's own smallest
// supported family, run by its c/olmoe.c). This package is the map from
// the HuggingFace checkpoint world to this engine's world:
//
//   - HFConfig reads the checkpoint's config.json with the exact keys
//     's olmoe.c reads (hidden_size, num_hidden_layers, ...);
//   - TensorMap names every HF tensor and where it lands in our layout —
//     the converter's ground truth;
//   - Template implements OLMoE-Instruct's real chat template;
//   - Plan() is the dry-run converter: given an HF checkpoint dir it
//     validates config + tensor inventory and prints the conversion and
//     memory plan ('s `coli plan`, adapter-scale).
//
// Honest status (mirrors the manual's Step 18): the mapping, template,
// and planner are complete and tested. Executing the conversion and
// running the result needs the four engine deltas the manual names —
// multi-head attention + RoPE, SwiGLU experts (three matrices), the
// real tokenizer via LoadMerges, per-group scales. Generate() says so
// rather than pretending.
package olmoe

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"algo-engine/models"
)

func init() { models.Register(Family{}) }

// ── HF config ────────────────────────────────────────────────────────────

// HFConfig mirrors the checkpoint's config.json. Field-for-field the
// keys 's olmoe.c req_num()s — kept identical so both projects
// read the same file the same way.
type HFConfig struct {
	HiddenSize        int     `json:"hidden_size"`         // d_model (2048)
	NumHiddenLayers   int     `json:"num_hidden_layers"`   // 16
	NumAttentionHeads int     `json:"num_attention_heads"` // 16
	NumKeyValueHeads  int     `json:"num_key_value_heads"` // 16
	NumExperts        int     `json:"num_experts"`         // 64
	NumExpertsPerTok  int     `json:"num_experts_per_tok"` // top-k = 8
	IntermediateSize  int     `json:"intermediate_size"`   // d_ff per expert (1024)
	VocabSize         int     `json:"vocab_size"`          // 50304
	RopeTheta         float64 `json:"rope_theta"`          // RoPE base (default 10000)
	RMSNormEps        float64 `json:"rms_norm_eps"`        // norm epsilon
	NormTopKProb      bool    `json:"norm_topk_prob"`      // renormalize top-k weights
}

// LoadHFConfig reads and sanity-checks a checkpoint's config.json.
func LoadHFConfig(hfDir string) (HFConfig, error) {
	raw, err := os.ReadFile(filepath.Join(hfDir, "config.json"))
	if err != nil {
		return HFConfig{}, fmt.Errorf("read HF config: %w", err)
	}
	var c HFConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return HFConfig{}, fmt.Errorf("parse HF config: %w", err)
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 10000 // 's default when absent
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 ||
		c.NumExperts < c.NumExpertsPerTok || c.NumExpertsPerTok <= 0 {
		return HFConfig{}, fmt.Errorf("HF config fails sanity checks: %+v", c)
	}
	return c, nil
}

// ── Tensor map ───────────────────────────────────────────────────────────

// Mapping is one row of the conversion table.
type Mapping struct {
	Ours     string // our layout's tensor name ("%d" = layer, "%d.%d" = layer.expert)
	HF       string // HF checkpoint name, same placeholders
	PerLayer bool
	PerExprt bool
	Dest     string // "dense" | "experts"
}

// TensorMap is the converter's ground truth: every tensor an OLMoE
// checkpoint holds and where it lands here. Names verified against
// 's olmoe.c loader and its convert_olmoe_merged.py (which merges
// each expert's gate/up/down into one record — the same adjacency trick
// as our expert.L.E layout, so "one pread per expert" survives).
var TensorMap = []Mapping{
	{Ours: "embed", HF: "model.embed_tokens.weight", Dest: "dense"},
	{Ours: "lm_head", HF: "lm_head.weight", Dest: "dense"}, // NOT tied in OLMoE
	{Ours: "norm_final", HF: "model.norm.weight", Dest: "dense"},

	{Ours: "layer.%d.norm_attn", HF: "model.layers.%d.input_layernorm.weight", PerLayer: true, Dest: "dense"},
	{Ours: "layer.%d.norm_ffn", HF: "model.layers.%d.post_attention_layernorm.weight", PerLayer: true, Dest: "dense"},
	{Ours: "layer.%d.wq", HF: "model.layers.%d.self_attn.q_proj.weight", PerLayer: true, Dest: "dense"},
	{Ours: "layer.%d.wk", HF: "model.layers.%d.self_attn.k_proj.weight", PerLayer: true, Dest: "dense"},
	{Ours: "layer.%d.wv", HF: "model.layers.%d.self_attn.v_proj.weight", PerLayer: true, Dest: "dense"},
	{Ours: "layer.%d.wo", HF: "model.layers.%d.self_attn.o_proj.weight", PerLayer: true, Dest: "dense"},
	// OLMoE adds q/k norms — new dense slots, no toy counterpart:
	{Ours: "layer.%d.q_norm", HF: "model.layers.%d.self_attn.q_norm.weight", PerLayer: true, Dest: "dense"},
	{Ours: "layer.%d.k_norm", HF: "model.layers.%d.self_attn.k_norm.weight", PerLayer: true, Dest: "dense"},
	{Ours: "layer.%d.gate", HF: "model.layers.%d.mlp.gate.weight", PerLayer: true, Dest: "dense"},

	// SwiGLU experts: THREE matrices per expert (toy has two). Our record
	// becomes [gate|up|down] int8 back-to-back + 3 scales — adjacency kept.
	{Ours: "expert.%d.%d.gate_proj", HF: "model.layers.%d.mlp.experts.%d.gate_proj.weight", PerExprt: true, Dest: "experts"},
	{Ours: "expert.%d.%d.up_proj", HF: "model.layers.%d.mlp.experts.%d.up_proj.weight", PerExprt: true, Dest: "experts"},
	{Ours: "expert.%d.%d.down_proj", HF: "model.layers.%d.mlp.experts.%d.down_proj.weight", PerExprt: true, Dest: "experts"},
}

// ExpandAll returns every concrete (ours, hf) pair for a config —
// the full conversion worklist.
func ExpandAll(c HFConfig) [][2]string {
	var out [][2]string
	for _, m := range TensorMap {
		switch {
		case m.PerExprt:
			for l := 0; l < c.NumHiddenLayers; l++ {
				for e := 0; e < c.NumExperts; e++ {
					out = append(out, [2]string{
						fmt.Sprintf(m.Ours, l, e), fmt.Sprintf(m.HF, l, e)})
				}
			}
		case m.PerLayer:
			for l := 0; l < c.NumHiddenLayers; l++ {
				out = append(out, [2]string{
					fmt.Sprintf(m.Ours, l), fmt.Sprintf(m.HF, l)})
			}
		default:
			out = append(out, [2]string{m.Ours, m.HF})
		}
	}
	return out
}

// ── Memory plan ──────────────────────────────────────────────────────────

// Plan is the dry-run converter's output: what conversion will produce
// and what running it will cost — 's `coli plan`, adapter-scale.
type Plan struct {
	Cfg          HFConfig
	Tensors      int
	DenseBytesI8 int64 // attention + norms + gates + embed, int8
	ExpertsBytes int64 // all experts, int8 (+ scales, negligible)
	PerExpert    int64 // one expert record (the streaming unit)
	ActiveSet    int64 // experts touched per token: layers × top-k
}

// BuildPlan computes the plan from a config (no weights needed).
func BuildPlan(c HFConfig) Plan {
	d, ff := int64(c.HiddenSize), int64(c.IntermediateSize)
	perExpert := 3 * d * ff // gate+up+down, int8 = 1 byte/weight
	experts := int64(c.NumHiddenLayers) * int64(c.NumExperts) * perExpert
	dense := int64(c.VocabSize)*d*2 + // embed + untied lm_head
		int64(c.NumHiddenLayers)*(4*d*d+ // q,k,v,o
			int64(c.NumExperts)*d+ // gate
			6*d) + d // norms (attn,ffn,q,k ×d each… kept coarse) + final
	return Plan{
		Cfg: c, Tensors: len(ExpandAll(c)),
		DenseBytesI8: dense, ExpertsBytes: experts, PerExpert: perExpert,
		ActiveSet: int64(c.NumHiddenLayers) * int64(c.NumExpertsPerTok) * perExpert,
	}
}

func (p Plan) String() string {
	gb := func(b int64) float64 { return float64(b) / (1 << 30) }
	mb := func(b int64) float64 { return float64(b) / (1 << 20) }
	return fmt.Sprintf(
		"OLMoE plan: %d layers × %d experts (top-%d), d=%d, d_ff=%d, %d heads\n"+
			"  tensors to convert : %d\n"+
			"  dense set (int8)   : %.2f GB  → RAM-resident\n"+
			"  experts (int8)     : %.2f GB  → disk, streamed (%.2f MB per expert)\n"+
			"  per-token active   : %.0f MB  (layers × top-k experts)\n"+
			"  suggested RAM tier : %.1f GB  (≈2× active set for >90%% hit rates)",
		p.Cfg.NumHiddenLayers, p.Cfg.NumExperts, p.Cfg.NumExpertsPerTok,
		p.Cfg.HiddenSize, p.Cfg.IntermediateSize, p.Cfg.NumAttentionHeads,
		p.Tensors, gb(p.DenseBytesI8), gb(p.ExpertsBytes), mb(p.PerExpert),
		mb(p.ActiveSet), 2*gb(p.ActiveSet))
}

// ── ModelEngine ──────────────────────────────────────────────────────────

// Family implements models.ModelEngine for OLMoE — mapping, template and
// planning complete; Generate gated on the Step 18 engine deltas.
type Family struct{}

func (Family) Family() string { return "olmoe" }

// Detect: an HF OLMoE checkpoint dir (config.json with model_type olmoe,
// or our converted layout once the converter runs).
func (Family) Detect(modelDir string) bool {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return false
	}
	var probe struct {
		ModelType string `json:"model_type"`
		Name      string `json:"name"`
	}
	if json.Unmarshal(raw, &probe) != nil {
		return false
	}
	return probe.ModelType == "olmoe" || probe.Name == "olmoe"
}

func (Family) Describe(modelDir string) (string, error) {
	c, err := LoadHFConfig(modelDir)
	if err != nil {
		return "", err
	}
	p := BuildPlan(c)
	return p.String(), nil
}

func (Family) Template() models.ChatTemplate { return Template{} }

// Template implements OLMoE-Instruct's real chat template, verbatim from
// tokenizer_config.json (documented in 's olmoe.c):
//
//	{bos}<|user|>\n{msg}\n<|assistant|>\n{reply}{eos}\n<|user|>\n...
//
// <|user|>/<|assistant|> are plain text (NOT special tokens); bos/eos are.
// The engine emits bos itself at sequence start, so Render starts at the
// first <|user|>.
type Template struct{}

func (Template) Render(msgs []models.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case "assistant":
			fmt.Fprintf(&b, "%s\n", m.Content) // closed by next turn's cue
		default: // user + system share the user slot in OLMoE's template
			fmt.Fprintf(&b, "<|user|>\n%s\n<|assistant|>\n", m.Content)
		}
	}
	return b.String()
}

func (Template) Stop() []string { return []string{"<|user|>"} }

// Generate is honestly gated: the four engine deltas come first.
func (Family) Generate(modelDir, session, prompt string, maxNew int,
	temperature float64, onToken func(text string)) (string, error) {
	return "", fmt.Errorf("olmoe: engine deltas pending (multi-head attention + RoPE, " +
		"SwiGLU 3-matrix experts, LoadMerges tokenizer, per-group scales) — " +
		"see Build Manual Step 18; mapping/template/plan are ready (try `algo-engine plan`)")
}
