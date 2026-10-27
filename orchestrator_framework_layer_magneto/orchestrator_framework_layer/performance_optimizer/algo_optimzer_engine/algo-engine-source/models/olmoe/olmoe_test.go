package olmoe

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"algo-engine/models"
)

// olmoe1b7bConfig is the real OLMoE-1B-7B shape (per the HF checkpoint,
// the same numbers 's olmoe.c reads).
const olmoe1b7bConfig = `{
  "model_type": "olmoe",
  "hidden_size": 2048,
  "num_hidden_layers": 16,
  "num_attention_heads": 16,
  "num_key_value_heads": 16,
  "num_experts": 64,
  "num_experts_per_tok": 8,
  "intermediate_size": 1024,
  "vocab_size": 50304,
  "rope_theta": 10000.0,
  "rms_norm_eps": 1e-05,
  "norm_topk_prob": false
}`

func writeCfg(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"),
		[]byte(olmoe1b7bConfig), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestLoadHFConfig(t *testing.T) {
	c, err := LoadHFConfig(writeCfg(t))
	if err != nil {
		t.Fatal(err)
	}
	if c.HiddenSize != 2048 || c.NumExperts != 64 || c.NumExpertsPerTok != 8 {
		t.Fatalf("config wrong: %+v", c)
	}
}

func TestTensorMapCovers1B7B(t *testing.T) {
	c, _ := LoadHFConfig(writeCfg(t))
	pairs := ExpandAll(c)

	// 16 layers × 64 experts × 3 projections + per-layer 9 + 3 globals
	wantExperts := 16 * 64 * 3
	wantDense := 16*9 + 3
	if len(pairs) != wantExperts+wantDense {
		t.Fatalf("worklist %d, want %d", len(pairs), wantExperts+wantDense)
	}
	// spot checks against 's verified HF names
	joined := ""
	for _, p := range pairs {
		joined += p[1] + "\n"
	}
	for _, must := range []string{
		"model.embed_tokens.weight",
		"model.layers.15.self_attn.q_norm.weight",
		"model.layers.0.mlp.gate.weight",
		"model.layers.15.mlp.experts.63.down_proj.weight",
	} {
		if !strings.Contains(joined, must) {
			t.Fatalf("worklist missing %s", must)
		}
	}
}

func TestPlanNumbers(t *testing.T) {
	c, _ := LoadHFConfig(writeCfg(t))
	p := BuildPlan(c)

	// one expert = 3 × 2048 × 1024 int8 = 6 MiB exactly
	if p.PerExpert != 3*2048*1024 {
		t.Fatalf("per-expert %d", p.PerExpert)
	}
	// experts total = 16 × 64 × 6 MiB = 6 GiB
	if p.ExpertsBytes != 16*64*p.PerExpert {
		t.Fatalf("experts %d", p.ExpertsBytes)
	}
	// active set per token = 16 layers × 8 experts × 6 MiB = 768 MiB
	if p.ActiveSet != 16*8*p.PerExpert {
		t.Fatalf("active %d", p.ActiveSet)
	}
	if !strings.Contains(p.String(), "6.00 GB") {
		t.Fatalf("plan text: %s", p)
	}
}

func TestDetectAndRegistry(t *testing.T) {
	if _, err := models.Get("olmoe"); err != nil {
		t.Fatal(err)
	}
	m, err := models.Open(writeCfg(t))
	if err != nil || m.Family() != "olmoe" {
		t.Fatalf("detect: %v %v", err, m)
	}
}

func TestTemplateMatchesOLMoEInstruct(t *testing.T) {
	got := (Template{}).Render([]models.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
		{Role: "user", Content: "bye"},
	})
	want := "<|user|>\nhi\n<|assistant|>\nhello\n<|user|>\nbye\n<|assistant|>\n"
	if got != want {
		t.Fatalf("template:\n got %q\nwant %q", got, want)
	}
	if (Template{}).Stop()[0] != "<|user|>" {
		t.Fatal("stop string")
	}
}

func TestGenerateHonestlyGated(t *testing.T) {
	_, err := (Family{}).Generate(writeCfg(t), "", "x", 1, 0, nil)
	if err == nil || !strings.Contains(err.Error(), "Step 18") {
		t.Fatalf("must gate with a pointer to the manual, got %v", err)
	}
}
