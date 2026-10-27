package toy

import (
	"strings"
	"testing"

	"algo-engine/internal/engine"
	"algo-engine/models"
)

func TestRegisteredAndDetects(t *testing.T) {
	if _, err := models.Get("toy"); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := engine.GenerateToyModel(dir, 1); err != nil {
		t.Fatal(err)
	}
	m, err := models.Open(dir) // auto-detect
	if err != nil {
		t.Fatal(err)
	}
	if m.Family() != "toy" {
		t.Fatalf("detected %q", m.Family())
	}
	desc, err := m.Describe(dir)
	if err != nil || !strings.Contains(desc, "toy-moe") {
		t.Fatalf("describe: %v %q", err, desc)
	}
}

func TestGenerateThroughInterface(t *testing.T) {
	dir := t.TempDir()
	if err := engine.GenerateToyModel(dir, 2); err != nil {
		t.Fatal(err)
	}
	m, _ := models.Get("toy")

	prompt := m.Template().Render([]models.Message{
		{Role: "user", Content: "hello"},
	})
	if !strings.HasSuffix(prompt, "assistant:") {
		t.Fatalf("template cue missing: %q", prompt)
	}

	streamed := 0
	out, err := m.Generate(dir, "", prompt, 8, 0, func(string) { streamed++ })
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 || streamed == 0 {
		t.Fatalf("no output through the interface (out=%d streamed=%d)",
			len(out), streamed)
	}
}
