// Package heatmap records which experts each workload actually routes to.
// It is the engine's memory of *your* usage: counts persist to usage.json,
// and the hottest experts earn permanent RAM residency (the "learned pin
// set") even when a strict LRU would have evicted them — the same idea as
// 's .coli_usage. The engine literally gets faster the more you
// run it on the same kind of work.
package heatmap

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"sync"
)

// Key identifies an expert globally: layer plus expert index.
type Key struct {
	Layer  int `json:"layer"`
	Expert int `json:"expert"`
}

// HeatMap is shared state: Route reads it, Learn writes it, and the API's
// stats endpoint reads it — hence the mutex.
type HeatMap struct {
	mu     sync.Mutex
	counts map[Key]int64
	hits   int64 // RAM-tier hits observed
	misses int64 // disk reads observed
}

func New() *HeatMap { return &HeatMap{counts: map[Key]int64{}} }

// Record notes one routing decision and whether it was a cache hit.
func (h *HeatMap) Record(k Key, hit bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.counts[k]++
	if hit {
		h.hits++
	} else {
		h.misses++
	}
}

// Stats returns cumulative hit/miss counters.
func (h *HeatMap) Stats() (hits, misses int64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.hits, h.misses
}

// Hottest returns the n most-routed experts, hottest first.
func (h *HeatMap) Hottest(n int) []Key {
	h.mu.Lock()
	defer h.mu.Unlock()
	keys := make([]Key, 0, len(h.counts))
	for k := range h.counts {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(a, b int) bool {
		ca, cb := h.counts[keys[a]], h.counts[keys[b]]
		if ca != cb {
			return ca > cb
		}
		if keys[a].Layer != keys[b].Layer {
			return keys[a].Layer < keys[b].Layer
		}
		return keys[a].Expert < keys[b].Expert
	})
	if n > len(keys) {
		n = len(keys)
	}
	return keys[:n]
}

// ── Persistence ──────────────────────────────────────────────────────────

type persisted struct {
	Counts []entry `json:"counts"`
}
type entry struct {
	Key   Key   `json:"key"`
	Count int64 `json:"count"`
}

// Save writes usage counts to path (typically usage.json).
func (h *HeatMap) Save(path string) error {
	h.mu.Lock()
	p := persisted{}
	for k, c := range h.counts {
		p.Counts = append(p.Counts, entry{k, c})
	}
	h.mu.Unlock()
	sort.Slice(p.Counts, func(a, b int) bool { // stable files diff nicely
		return p.Counts[a].Count > p.Counts[b].Count
	})
	raw, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

// Load merges previously saved counts (missing file is not an error —
// a fresh machine just starts cold).
func (h *HeatMap) Load(path string) error {
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	var p persisted
	if err := json.Unmarshal(raw, &p); err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, e := range p.Counts {
		h.counts[e.Key] += e.Count
	}
	return nil
}
