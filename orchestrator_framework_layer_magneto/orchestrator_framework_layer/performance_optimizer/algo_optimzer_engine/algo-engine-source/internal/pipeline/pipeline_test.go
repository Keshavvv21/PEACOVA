package pipeline

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"algo-engine/internal/expertstore"
	"algo-engine/internal/heatmap"
	"algo-engine/internal/quant"
	"algo-engine/internal/router"
	"algo-engine/internal/safetensors"
)

// buildStore writes a 1-layer, 4-expert shard where expert e's network
// computes: W1 = identity-ish rows scaled (e+1), W2 = transpose pattern,
// simple enough to hand-verify.
func buildStore(t *testing.T) (*expertstore.Store, expertstore.Layout) {
	t.Helper()
	layout := expertstore.Layout{DModel: 2, DFF: 2}
	w := safetensors.NewWriter()
	for e := 0; e < 4; e++ {
		// W1 = [[e+1,0],[0,e+1]]  W2 = [[1,0],[0,1]]
		s := float32(e + 1)
		w1 := []float32{s, 0, 0, s}
		w2 := []float32{1, 0, 0, 1}
		q1, s1 := quant.Quantize8(w1)
		q2, s2 := quant.Quantize8(w2)
		raw := append(quant.BytesFromInt8(q1), quant.BytesFromInt8(q2)...)
		name := "expert.0." + string(rune('0'+e))
		w.AddI8(name, []int64{int64(len(raw))}, raw)
		w.AddF32(name+".scales", []int64{2}, []float32{s1, s2})
	}
	path := filepath.Join(t.TempDir(), "e.safetensors")
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return expertstore.New(f, layout, heatmap.New(),
		expertstore.Config{Capacity: 2, MaxConcurrentReads: 2}), layout
}

func TestForwardBlendsExperts(t *testing.T) {
	store, layout := buildStore(t)
	m := &MoE{Store: store, Kernel: quant.Pure{}, Layout: layout}

	// Experts 0 (scale 1) and 2 (scale 3), weights 0.25 / 0.75, x=[1,1]:
	// y = 0.25*[1,1] + 0.75*[3,3] = [2.5, 2.5]
	choices := []router.Choice{
		{ID: 0, Weight: 0.25},
		{ID: 2, Weight: 0.75},
	}
	y, err := m.Forward(context.Background(), 0, []float32{1, 1}, choices)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range y {
		if math.Abs(float64(v)-2.5) > 0.02 { // int8 tolerance
			t.Fatalf("y[%d]=%v want ~2.5", i, v)
		}
	}
}

func TestForwardDeterministic(t *testing.T) {
	store, layout := buildStore(t)
	m := &MoE{Store: store, Kernel: quant.Pure{}, Layout: layout}
	choices := []router.Choice{{ID: 1, Weight: 0.5}, {ID: 3, Weight: 0.5}}
	x := []float32{0.3, -0.7}

	first, err := m.Forward(context.Background(), 0, x, choices)
	if err != nil {
		t.Fatal(err)
	}
	for run := 0; run < 20; run++ { // goroutine scheduling must not matter
		y, err := m.Forward(context.Background(), 0, x, choices)
		if err != nil {
			t.Fatal(err)
		}
		for i := range y {
			if y[i] != first[i] {
				t.Fatalf("run %d diverged at %d: %v vs %v", run, i, y[i], first[i])
			}
		}
	}
}
