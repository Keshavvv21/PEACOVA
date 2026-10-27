package heatmap

import (
	"path/filepath"
	"testing"
)

func TestHottestOrdersByCount(t *testing.T) {
	h := New()
	hot := Key{Layer: 1, Expert: 7}
	warm := Key{Layer: 0, Expert: 3}
	for i := 0; i < 5; i++ {
		h.Record(hot, true)
	}
	h.Record(warm, false)

	top := h.Hottest(2)
	if top[0] != hot || top[1] != warm {
		t.Fatalf("got %v", top)
	}
	hits, misses := h.Stats()
	if hits != 5 || misses != 1 {
		t.Fatalf("hits=%d misses=%d", hits, misses)
	}
}

func TestSaveLoadMerges(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.json")
	h := New()
	k := Key{Layer: 2, Expert: 4}
	h.Record(k, true)
	if err := h.Save(path); err != nil {
		t.Fatal(err)
	}

	h2 := New()
	h2.Record(k, true) // one fresh count...
	if err := h2.Load(path); err != nil {
		t.Fatal(err)
	}
	h2.Record(k, true) // ...plus one more after load
	top := h2.Hottest(1)
	if len(top) != 1 || top[0] != k {
		t.Fatalf("got %v", top)
	}
}

func TestLoadMissingFileIsFine(t *testing.T) {
	if err := New().Load(filepath.Join(t.TempDir(), "absent.json")); err != nil {
		t.Fatal(err)
	}
}
