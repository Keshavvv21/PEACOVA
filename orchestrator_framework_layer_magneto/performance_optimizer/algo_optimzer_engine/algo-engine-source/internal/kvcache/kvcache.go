// Package kvcache stores the attention keys and values of every past
// token, so generating token N costs one attention pass over cached
// state instead of recomputing the whole prefix. It also persists to
// disk per session, so a conversation reopens "warm" — the same idea
// as 's .coli_kv (minus the MLA compression, which is a
// documented extension, not reproduced here).
package kvcache

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
)

// Cache holds per-layer K and V vectors for each generated position.
// Keys[layer][pos] is a dModel-length key vector (single-head toy model;
// a multi-head model shards the same idea per head).
type Cache struct {
	SessionID string
	NLayers   int
	Keys      [][][]float32
	Values    [][][]float32
}

func New(sessionID string, nLayers int) *Cache {
	return &Cache{
		SessionID: sessionID,
		NLayers:   nLayers,
		Keys:      make([][][]float32, nLayers),
		Values:    make([][][]float32, nLayers),
	}
}

// Append stores one position's key/value for a layer.
func (c *Cache) Append(layer int, k, v []float32) {
	c.Keys[layer] = append(c.Keys[layer], k)
	c.Values[layer] = append(c.Values[layer], v)
}

// Len is the number of cached positions (uniform across layers).
func (c *Cache) Len() int {
	if c.NLayers == 0 {
		return 0
	}
	return len(c.Keys[0])
}

// Layer returns the cached keys and values for one layer.
func (c *Cache) Layer(layer int) (ks, vs [][]float32) {
	return c.Keys[layer], c.Values[layer]
}

// ── Persistence ──────────────────────────────────────────────────────────

func sessionPath(dir, sessionID string) string {
	return filepath.Join(dir, sessionID+".kv")
}

// Save writes the cache with gob — a simple, honest baseline. A binary
// packed format (and compression) is the documented follow-up.
func (c *Cache) Save(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.Create(sessionPath(dir, c.SessionID))
	if err != nil {
		return fmt.Errorf("create kv file: %w", err)
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(c)
}

// Load restores a session's cache; a missing file returns a fresh cache
// (cold start is not an error).
func Load(dir, sessionID string, nLayers int) (*Cache, error) {
	f, err := os.Open(sessionPath(dir, sessionID))
	if os.IsNotExist(err) {
		return New(sessionID, nLayers), nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var c Cache
	if err := gob.NewDecoder(f).Decode(&c); err != nil {
		return nil, fmt.Errorf("decode kv cache: %w", err)
	}
	if c.NLayers != nLayers {
		return nil, fmt.Errorf("kv cache has %d layers, model has %d",
			c.NLayers, nLayers)
	}
	return &c, nil
}
