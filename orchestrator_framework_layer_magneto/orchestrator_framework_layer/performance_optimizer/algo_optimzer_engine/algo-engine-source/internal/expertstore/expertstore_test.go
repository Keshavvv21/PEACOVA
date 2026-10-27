package expertstore

import (
	"context"
	"path/filepath"
	"testing"

	"algo-engine/internal/heatmap"
	"algo-engine/internal/quant"
	"algo-engine/internal/safetensors"
)

// writeTestShard builds a safetensors file with nLayers×nExperts experts
// whose W1 weights are all (layer+1) and W2 all (id+1) — recognizable
// values that let tests verify the right bytes came back.
func writeTestShard(t *testing.T, path string, layout Layout, nLayers, nExperts int) {
	t.Helper()
	w := safetensors.NewWriter()
	for l := 0; l < nLayers; l++ {
		for e := 0; e < nExperts; e++ {
			w1 := make([]float32, layout.W1Len())
			w2 := make([]float32, layout.W2Len())
			for i := range w1 {
				w1[i] = float32(l + 1)
			}
			for i := range w2 {
				w2[i] = float32(e + 1)
			}
			q1, s1 := quant.Quantize8(w1)
			q2, s2 := quant.Quantize8(w2)
			raw := append(quant.BytesFromInt8(q1), quant.BytesFromInt8(q2)...)
			name := tensorName(l, e)
			w.AddI8(name, []int64{int64(len(raw))}, raw)
			w.AddF32(name+".scales", []int64{2}, []float32{s1, s2})
		}
	}
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
}

func openStore(t *testing.T, capacity int) (*Store, *heatmap.HeatMap) {
	t.Helper()
	layout := Layout{DModel: 4, DFF: 8}
	path := filepath.Join(t.TempDir(), "experts.safetensors")
	writeTestShard(t, path, layout, 2, 4)
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	heat := heatmap.New()
	return New(f, layout, heat, Config{Capacity: capacity, MaxConcurrentReads: 2}), heat
}

func TestMissThenHit(t *testing.T) {
	s, heat := openStore(t, 4)
	ctx := context.Background()

	e, hit, err := s.Get(ctx, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if hit {
		t.Fatal("first access must be a disk miss")
	}
	// W1 was all 2.0 (layer+1); int8 round trip is near-exact for constants
	if e.W1[0] < 1.99 || e.W1[0] > 2.01 {
		t.Fatalf("W1[0]=%v want ~2.0", e.W1[0])
	}
	if e.W2[0] < 2.99 || e.W2[0] > 3.01 { // id+1 = 3
		t.Fatalf("W2[0]=%v want ~3.0", e.W2[0])
	}

	_, hit, err = s.Get(ctx, 1, 2)
	if err != nil || !hit {
		t.Fatalf("second access must be a RAM hit (err=%v hit=%v)", err, hit)
	}
	hits, misses := heat.Stats()
	if hits != 1 || misses != 1 {
		t.Fatalf("hits=%d misses=%d", hits, misses)
	}
}

func TestLRUEvictsOldest(t *testing.T) {
	s, _ := openStore(t, 2)
	ctx := context.Background()

	s.Get(ctx, 0, 0)
	s.Get(ctx, 0, 1)
	s.Get(ctx, 0, 2) // capacity 2 -> evicts (0,0)

	if _, hit, _ := s.Get(ctx, 0, 0); hit {
		t.Fatal("(0,0) should have been evicted")
	}
	if s.Evicted() < 1 {
		t.Fatal("eviction counter should have advanced")
	}
}

func TestPinsSurviveEviction(t *testing.T) {
	s, _ := openStore(t, 1)
	ctx := context.Background()

	if err := s.Pin(ctx, []heatmap.Key{{Layer: 0, Expert: 3}}); err != nil {
		t.Fatal(err)
	}
	// churn the tiny LRU hard
	for i := 0; i < 3; i++ {
		s.Get(ctx, 0, i)
	}
	if _, hit, _ := s.Get(ctx, 0, 3); !hit {
		t.Fatal("pinned expert must never be evicted")
	}
}
