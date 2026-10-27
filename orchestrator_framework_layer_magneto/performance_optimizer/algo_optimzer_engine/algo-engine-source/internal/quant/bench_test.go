package quant

// Benchmarks answer the One Rule's empirical question — "is C actually
// faster HERE?" — with numbers instead of belief. Run both ways:
//
//   go test -bench=. -benchmem ./internal/quant/                  # oracle
//   go test -tags asmkernel -bench=. -benchmem ./internal/quant/  # cgo
//
// and compare ns/op at the model's real shapes before shipping the tag.

import (
	"math/rand"
	"testing"
)

func benchInputs(m, n int) ([]float32, []int8, float32, []float32) {
	rng := rand.New(rand.NewSource(7))
	w := make([]float32, m*n)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q, s := Quantize8(w)
	return w, q, s, x
}

func BenchmarkMatVec128x64(b *testing.B) {
	w, _, _, x := benchInputs(128, 64)
	k := Active()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k.MatVec(w, 128, 64, x)
	}
}

func BenchmarkMatVecQ8_128x64(b *testing.B) {
	_, q, s, x := benchInputs(128, 64)
	k := Active()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k.MatVecQ8(q, s, 128, 64, x)
	}
}
