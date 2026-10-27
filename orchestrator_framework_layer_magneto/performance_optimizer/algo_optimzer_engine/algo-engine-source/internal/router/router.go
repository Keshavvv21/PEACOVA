// Package router implements the MoE gate: a small linear layer that scores
// every expert for the current hidden state, followed by top-k selection
// and a softmax over just the winners. The chosen experts' outputs are
// blended with the softmax weights — so routing decides both *which*
// experts run and *how much* each one's answer counts.
package router

import (
	"math"
	"sort"

	"algo-engine/internal/quant"
)

// ExpertID identifies one expert within one layer.
type ExpertID int

// Choice is one selected expert with its blend weight.
type Choice struct {
	ID     ExpertID
	Weight float32 // softmax weight over the top-k, sums to 1 across choices
}

// Gate is the routing network for one layer: W is [nExperts × dModel].
type Gate struct {
	W        []float32
	NExperts int
	DModel   int
	TopK     int
	kernel   quant.Kernel
}

func NewGate(w []float32, nExperts, dModel, topK int) *Gate {
	return &Gate{W: w, NExperts: nExperts, DModel: dModel, TopK: topK,
		kernel: quant.Active()}
}

// Route scores all experts for hidden state h and returns the top-k with
// softmax blend weights. Deterministic: ties break by lower expert ID.
func (g *Gate) Route(h []float32) []Choice {
	logits := g.kernel.MatVec(g.W, g.NExperts, g.DModel, h)

	// indices sorted by logit descending, ID ascending on ties
	idx := make([]int, len(logits))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		if logits[idx[a]] != logits[idx[b]] {
			return logits[idx[a]] > logits[idx[b]]
		}
		return idx[a] < idx[b]
	})

	k := g.TopK
	if k > len(idx) {
		k = len(idx)
	}
	top := idx[:k]

	// numerically stable softmax over the k winners only
	maxL := logits[top[0]]
	var sum float64
	exps := make([]float64, k)
	for i, id := range top {
		e := math.Exp(float64(logits[id] - maxL))
		exps[i] = e
		sum += e
	}
	out := make([]Choice, k)
	for i, id := range top {
		out[i] = Choice{ID: ExpertID(id), Weight: float32(exps[i] / sum)}
	}
	return out
}
