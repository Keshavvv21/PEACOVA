// Package pipeline implements the five-step per-token MoE path from
// 's design — route → union → place → overlap → learn — as a Go
// concurrency pattern.
//
// For each MoE layer the engine asks: "run these routed experts on this
// input". The pipeline dedupes the requested experts (union), asks the
// store where each lives (place), then overlaps disk streaming of misses
// with compute on RAM hits (overlap): every expert gets its own goroutine,
// disk parallelism is bounded inside the store, and results are blended
// as they arrive. The learn step happens inside the store via the heat
// map on every Get.
//
// Because expert outputs are summed with router weights, arrival order
// does not matter — floating-point addition order is fixed by expert ID
// at the end so results stay bit-identical run to run.
package pipeline

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"algo-engine/internal/expertstore"
	"algo-engine/internal/quant"
	"algo-engine/internal/router"
)

// MoE executes routed expert computation for one layer.
type MoE struct {
	Store  *expertstore.Store
	Kernel quant.Kernel
	Layout expertstore.Layout
}

type result struct {
	id  router.ExpertID
	out []float32
	w   float32
	err error
}

// Forward runs the top-k experts on input x and returns the weighted sum
// of their outputs: y = Σ wᵢ · Expertᵢ(x).
//
//	route   — already done by the caller (choices)
//	union   — dedupe choices (a token can't route to the same expert twice
//	          in one layer, but batched callers can — kept general)
//	place+overlap — Store.Get inside per-expert goroutines: RAM hits
//	          compute immediately while misses stream from disk in parallel
//	learn   — Store.Get records every access in the heat map
func (m *MoE) Forward(ctx context.Context, layer int, x []float32,
	choices []router.Choice) ([]float32, error) {

	// union: dedupe, summing weights of duplicates
	weights := map[router.ExpertID]float32{}
	for _, c := range choices {
		weights[c.ID] += c.Weight
	}

	results := make(chan result, len(weights))
	var wg sync.WaitGroup
	for id, w := range weights {
		wg.Add(1)
		go func(id router.ExpertID, w float32) {
			defer wg.Done()
			e, _, err := m.Store.Get(ctx, layer, int(id))
			if err != nil {
				results <- result{id: id, err: err}
				return
			}
			results <- result{id: id, w: w, out: m.runExpert(e, x)}
		}(id, w)
	}
	wg.Wait()
	close(results)

	// deterministic accumulation: sort by expert ID before summing
	var done []result
	for r := range results {
		if r.err != nil {
			return nil, fmt.Errorf("expert %d layer %d: %w", r.id, layer, r.err)
		}
		done = append(done, r)
	}
	sort.Slice(done, func(a, b int) bool { return done[a].id < done[b].id })

	y := make([]float32, len(x))
	for _, r := range done {
		for i := range y {
			y[i] += r.w * r.out[i]
		}
	}
	return y, nil
}

// runExpert computes Expert(x) = W2 · relu(W1 · x).
func (m *MoE) runExpert(e *expertstore.Expert, x []float32) []float32 {
	h := m.Kernel.MatVec(e.W1, m.Layout.DFF, m.Layout.DModel, x)
	for i, v := range h {
		if v < 0 {
			h[i] = 0 // ReLU
		}
	}
	return m.Kernel.MatVec(e.W2, m.Layout.DModel, m.Layout.DFF, h)
}
