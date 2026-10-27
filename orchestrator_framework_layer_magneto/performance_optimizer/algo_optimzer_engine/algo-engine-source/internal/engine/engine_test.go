package engine

import (
	"context"
	"testing"

	"algo-engine/internal/kvcache"
)

func loadToy(t *testing.T) (*Engine, string) {
	t.Helper()
	dir := t.TempDir()
	if err := GenerateToyModel(dir, 42); err != nil {
		t.Fatal(err)
	}
	e, err := Load(dir, Options{CacheCapacity: 32, MaxConcurrentReads: 4})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close(dir) })
	return e, dir
}

func TestGenerateIsDeterministic(t *testing.T) {
	e, _ := loadToy(t)
	ctx := context.Background()

	run := func() []int {
		cache := kvcache.New("s", e.Cfg.NLayers)
		ids, err := e.Generate(ctx, "hello", 12,
			NewSampler(0, 0, 0), cache, nil) // greedy
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	a, b := run(), run()
	if len(a) == 0 {
		t.Fatal("no tokens generated")
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("greedy decode must be deterministic: %v vs %v", a, b)
		}
	}
}

func TestSameSeedSameModelSameOutput(t *testing.T) {
	// Two independently generated model dirs with the same seed must
	// produce identical generations — the reproducibility contract.
	ctx := context.Background()
	gen := func() []int {
		dir := t.TempDir()
		if err := GenerateToyModel(dir, 7); err != nil {
			t.Fatal(err)
		}
		e, err := Load(dir, Options{CacheCapacity: 16, MaxConcurrentReads: 2})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Close(dir)
		ids, err := e.Generate(ctx, "abc", 8, NewSampler(0, 0, 0),
			kvcache.New("s", e.Cfg.NLayers), nil)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	a, b := gen(), gen()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed diverged: %v vs %v", a, b)
		}
	}
}

func TestExpertStreamingHappens(t *testing.T) {
	e, _ := loadToy(t)
	_, err := e.Generate(context.Background(), "streaming test", 10,
		NewSampler(0, 0, 0), kvcache.New("s", e.Cfg.NLayers), nil)
	if err != nil {
		t.Fatal(err)
	}
	hits, misses := e.Heat.Stats()
	if misses == 0 {
		t.Fatal("expected disk misses on a cold cache")
	}
	if hits == 0 {
		t.Fatal("expected RAM hits once experts became resident")
	}
	if e.Store.ResidentCount() == 0 {
		t.Fatal("RAM tier should hold experts after a run")
	}
}

func TestTinyCacheStillCorrect(t *testing.T) {
	// Capacity 1 forces constant eviction and re-streaming; output must
	// match the big-cache run exactly — placement changes speed, never
	// semantics.
	ctx := context.Background()
	dir := t.TempDir()
	if err := GenerateToyModel(dir, 99); err != nil {
		t.Fatal(err)
	}
	gen := func(capacity int) []int {
		e, err := Load(dir, Options{CacheCapacity: capacity, MaxConcurrentReads: 2})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Close(dir)
		ids, err := e.Generate(ctx, "tiers", 8, NewSampler(0, 0, 0),
			kvcache.New("s", e.Cfg.NLayers), nil)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	big, tiny := gen(64), gen(1)
	for i := range big {
		if big[i] != tiny[i] {
			t.Fatalf("cache size changed OUTPUT (%v vs %v) — semantics violated", big, tiny)
		}
	}
}
