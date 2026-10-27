// Package models defines the multi-family seam from the Technical Design
// (§7): every model family — toy today, OLMoE next, others later — is one
// package implementing ModelEngine, and everything above this line (CLI,
// API, dashboard, expert store) stays family-agnostic. This is the Go
// translation of 's rule: "one C file per family, same coli
// chat/serve/web front end" — the file becomes a package, the shared
// headers become interfaces.
package models

import (
	"fmt"
	"sort"
	"sync"
)

// ChatTemplate renders a conversation into the exact prompt string a
// family was trained on. Getting this wrong doesn't crash — it quietly
// degrades quality, which is why it is part of the interface and not a
// helper in the API gateway.
type ChatTemplate interface {
	// Render formats the message history ending with the assistant cue.
	Render(msgs []Message) string
	// Stop returns strings whose generation should end the turn.
	Stop() []string
}

// Message mirrors the OpenAI wire shape the gateway already speaks.
type Message struct {
	Role    string
	Content string
}

// ModelEngine is what every family package implements — the contract the
// Technical Design specifies verbatim. A family that satisfies it gets
// the whole stack for free: tiered streaming, heat-map pins, KV
// persistence, the CLI, the API, the dashboard.
type ModelEngine interface {
	// Family is the registry name ("toy", "olmoe", ...).
	Family() string

	// Detect reports whether modelDir looks like this family's layout —
	// used by Open() to auto-select a family from a directory.
	Detect(modelDir string) bool

	// Describe returns a one-line human summary of the model in modelDir
	// (shape, quantization, disk footprint) without fully loading it.
	Describe(modelDir string) (string, error)

	// Template returns the family's chat template.
	Template() ChatTemplate

	// Generate runs the family's engine: prompt in, streamed tokens out
	// via onToken, generated text returned. Sessions with the same
	// non-empty session name continue warm (KV persistence).
	Generate(modelDir, session, prompt string, maxNew int,
		temperature float64, onToken func(text string)) (string, error)
}

// ── Registry ─────────────────────────────────────────────────────────────

var (
	mu       sync.Mutex
	registry = map[string]ModelEngine{}
)

// Register is called from each family package's init(). Importing the
// package IS enabling the family — the blank-import idiom, same as
// database/sql drivers (and Step 3's safetensors reader has no idea any
// of this exists, which is the point).
func Register(m ModelEngine) {
	mu.Lock()
	defer mu.Unlock()
	if _, dup := registry[m.Family()]; dup {
		panic("models: duplicate family " + m.Family())
	}
	registry[m.Family()] = m
}

// Get returns a family by name.
func Get(family string) (ModelEngine, error) {
	mu.Lock()
	defer mu.Unlock()
	m, ok := registry[family]
	if !ok {
		return nil, fmt.Errorf("unknown model family %q (have: %v)", family, names())
	}
	return m, nil
}

// Open auto-detects the family for a model directory.
func Open(modelDir string) (ModelEngine, error) {
	mu.Lock()
	defer mu.Unlock()
	for _, name := range names() {
		if registry[name].Detect(modelDir) {
			return registry[name], nil
		}
	}
	return nil, fmt.Errorf("no registered family recognizes %s (have: %v)",
		modelDir, names())
}

// List returns registered family names, sorted.
func List() []string {
	mu.Lock()
	defer mu.Unlock()
	return names()
}

func names() []string {
	out := make([]string, 0, len(registry))
	for k := range registry {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
