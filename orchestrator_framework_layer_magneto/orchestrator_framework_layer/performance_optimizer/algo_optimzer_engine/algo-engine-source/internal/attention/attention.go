// Package attention implements single-head causal self-attention with a
// KV cache — the "memory" half of every transformer layer.
//
// For the new token's hidden state h:
//
//	q = Wq·h    k = Wk·h    v = Wv·h
//	append (k, v) to the cache
//	scores_i = (q · k_i) / sqrt(d)      for every cached position i
//	weights  = softmax(scores)          (causal: only past+current exist)
//	ctx      = Σ weights_i · v_i
//	out      = Wo·ctx
//
// Production note: real models split q/k/v into many heads that attend
// independently and concatenate — same math per head, plus rotary
// position embeddings (RoPE) on q and k. The single-head version keeps
// every concept and drops only the bookkeeping.
package attention

import (
	"math"

	"algo-engine/internal/kvcache"
	"algo-engine/internal/quant"
)

// Layer is one attention block's weights, each [DModel × DModel] row-major.
type Layer struct {
	Wq, Wk, Wv, Wo []float32
	DModel         int
	kernel         quant.Kernel
}

func NewLayer(wq, wk, wv, wo []float32, dModel int) *Layer {
	return &Layer{Wq: wq, Wk: wk, Wv: wv, Wo: wo, DModel: dModel,
		kernel: quant.Active()}
}

// Forward runs attention for one new token, reading and extending the
// KV cache for layerIdx.
func (l *Layer) Forward(h []float32, cache *kvcache.Cache, layerIdx int) []float32 {
	d := l.DModel
	q := l.kernel.MatVec(l.Wq, d, d, h)
	k := l.kernel.MatVec(l.Wk, d, d, h)
	v := l.kernel.MatVec(l.Wv, d, d, h)

	cache.Append(layerIdx, k, v)
	ks, vs := cache.Layer(layerIdx)

	// scaled dot-product scores against every cached position
	scale := 1 / math.Sqrt(float64(d))
	scores := make([]float64, len(ks))
	maxS := math.Inf(-1)
	for i, ki := range ks {
		var dot float64
		for j := range ki {
			dot += float64(q[j]) * float64(ki[j])
		}
		scores[i] = dot * scale
		if scores[i] > maxS {
			maxS = scores[i]
		}
	}

	// stable softmax
	var sum float64
	for i := range scores {
		scores[i] = math.Exp(scores[i] - maxS)
		sum += scores[i]
	}

	// weighted sum of values
	ctxVec := make([]float32, d)
	for i, vi := range vs {
		w := float32(scores[i] / sum)
		for j := range vi {
			ctxVec[j] += w * vi[j]
		}
	}
	return l.kernel.MatVec(l.Wo, d, d, ctxVec)
}
