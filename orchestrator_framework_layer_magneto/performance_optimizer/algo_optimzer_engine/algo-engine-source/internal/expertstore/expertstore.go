// Package expertstore is the tiered expert cache — the heart of the
//
//	idea in Go. Expert weights live on disk inside a safetensors
//
// file; a bounded RAM tier holds the hot set (LRU eviction plus a learned
// pin set from the heat map); Get returns an expert's dequantized weights
// from RAM when possible and streams from disk when not, with disk
// concurrency bounded by a semaphore channel.
//
// The tier an expert answers from may only ever change speed — never the
// numbers. Both paths dequantize the identical bytes with the identical
// scale.
package expertstore

import (
	"container/list"
	"context"
	"fmt"
	"sync"

	"algo-engine/internal/heatmap"
	"algo-engine/internal/quant"
	"algo-engine/internal/safetensors"
)

// Expert is one expert's ready-to-use weights: a two-layer feed-forward
// network y = W2·relu(W1·x), stored dequantized for compute.
type Expert struct {
	Layer int
	ID    int
	W1    []float32 // [DFF × DModel] row-major
	W2    []float32 // [DModel × DFF] row-major
}

// Layout describes where experts live inside the safetensors file. Each
// expert is one I8 tensor named "expert.<layer>.<id>" holding W1 then W2
// back to back, plus one F32 tensor "expert.<layer>.<id>.scales" holding
// [scale1, scale2]. Fixed shapes mean zero per-expert metadata lookups.
type Layout struct {
	DModel, DFF int
}

func (l Layout) W1Len() int  { return l.DFF * l.DModel }
func (l Layout) W2Len() int  { return l.DModel * l.DFF }
func (l Layout) RecLen() int { return l.W1Len() + l.W2Len() }

func tensorName(layer, id int) string { return fmt.Sprintf("expert.%d.%d", layer, id) }

// Store is the tiered cache.
type Store struct {
	file   *safetensors.File
	layout Layout
	heat   *heatmap.HeatMap

	mu       sync.Mutex
	resident map[heatmap.Key]*Expert       // RAM tier (pins + LRU entries)
	lru      *list.List                    // front = most recent
	lruPos   map[heatmap.Key]*list.Element // key -> its LRU node
	pinned   map[heatmap.Key]bool          // learned pins: never evicted
	capacity int                           // max experts in RAM (pins excluded)

	inflight chan struct{} // bounds concurrent disk reads
	dropped  int64
}

// Config for the store.
type Config struct {
	Capacity           int // RAM tier size in experts (beyond pins)
	MaxConcurrentReads int // disk read parallelism
}

func New(file *safetensors.File, layout Layout, heat *heatmap.HeatMap, cfg Config) *Store {
	if cfg.Capacity < 1 {
		cfg.Capacity = 1
	}
	if cfg.MaxConcurrentReads < 1 {
		cfg.MaxConcurrentReads = 4
	}
	return &Store{
		file: file, layout: layout, heat: heat,
		resident: map[heatmap.Key]*Expert{},
		lru:      list.New(),
		lruPos:   map[heatmap.Key]*list.Element{},
		pinned:   map[heatmap.Key]bool{},
		capacity: cfg.Capacity,
		inflight: make(chan struct{}, cfg.MaxConcurrentReads),
	}
}

// Get returns the expert's weights, from RAM if resident, else from disk.
// The bool reports whether it was a RAM hit (for stats and the heat map).
func (s *Store) Get(ctx context.Context, layer, id int) (*Expert, bool, error) {
	k := heatmap.Key{Layer: layer, Expert: id}

	s.mu.Lock()
	if e, ok := s.resident[k]; ok {
		s.touchLocked(k)
		s.mu.Unlock()
		s.heat.Record(k, true)
		return e, true, nil
	}
	s.mu.Unlock()

	// Miss: bounded disk read. The semaphore keeps a slow disk from
	// accumulating unbounded concurrent reads.
	select {
	case s.inflight <- struct{}{}:
		defer func() { <-s.inflight }()
	case <-ctx.Done():
		return nil, false, ctx.Err()
	}

	e, err := s.load(layer, id)
	if err != nil {
		return nil, false, err
	}
	s.heat.Record(k, false)

	s.mu.Lock()
	s.insertLocked(k, e)
	s.mu.Unlock()
	return e, false, nil
}

// load streams one expert from the safetensors file and dequantizes it.
func (s *Store) load(layer, id int) (*Expert, error) {
	name := tensorName(layer, id)

	raw := make([]byte, s.layout.RecLen())
	if err := s.file.ReadRawAt(name, 0, int64(len(raw)), raw); err != nil {
		return nil, fmt.Errorf("stream %s: %w", name, err)
	}
	scales, err := s.file.ReadF32(name + ".scales")
	if err != nil {
		return nil, fmt.Errorf("scales for %s: %w", name, err)
	}
	if len(scales) != 2 {
		return nil, fmt.Errorf("%s: want 2 scales, got %d", name, len(scales))
	}

	w1q := quant.Int8FromBytes(raw[:s.layout.W1Len()])
	w2q := quant.Int8FromBytes(raw[s.layout.W1Len():])
	return &Expert{
		Layer: layer, ID: id,
		W1: quant.Dequant8(w1q, scales[0]),
		W2: quant.Dequant8(w2q, scales[1]),
	}, nil
}

// ── RAM tier bookkeeping (caller holds s.mu) ─────────────────────────────

func (s *Store) touchLocked(k heatmap.Key) {
	if el, ok := s.lruPos[k]; ok {
		s.lru.MoveToFront(el)
	}
}

func (s *Store) insertLocked(k heatmap.Key, e *Expert) {
	if _, ok := s.resident[k]; ok {
		s.touchLocked(k)
		return
	}
	s.resident[k] = e
	if s.pinned[k] {
		return // pins live outside the LRU
	}
	s.lruPos[k] = s.lru.PushFront(k)
	for s.lru.Len() > s.capacity {
		oldest := s.lru.Back()
		ok := oldest.Value.(heatmap.Key)
		s.lru.Remove(oldest)
		delete(s.lruPos, ok)
		delete(s.resident, ok)
		s.dropped++
	}
}

// Pin marks experts as permanently resident and loads them now (warmup).
// Called at startup with the heat map's hottest experts.
func (s *Store) Pin(ctx context.Context, keys []heatmap.Key) error {
	for _, k := range keys {
		s.mu.Lock()
		s.pinned[k] = true
		// if it is currently an LRU entry, promote it out of the LRU
		if el, ok := s.lruPos[k]; ok {
			s.lru.Remove(el)
			delete(s.lruPos, k)
		}
		already := s.resident[k] != nil
		s.mu.Unlock()
		if already {
			continue
		}
		e, err := s.load(k.Layer, k.Expert)
		if err != nil {
			return fmt.Errorf("pin warmup %v: %w", k, err)
		}
		s.mu.Lock()
		s.resident[k] = e
		s.mu.Unlock()
	}
	return nil
}

// ResidentCount reports RAM-tier occupancy (pins + LRU entries).
func (s *Store) ResidentCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.resident)
}

// Evicted reports how many experts the LRU has pushed out.
func (s *Store) Evicted() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dropped
}
