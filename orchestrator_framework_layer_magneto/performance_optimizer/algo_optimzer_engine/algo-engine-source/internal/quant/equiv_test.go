//go:build asmkernel || cuda

package quant

// The guard on the triangle's tip: an accelerated kernel may only ever
// change speed. This test compares Active() (the cgo kernel in this
// build) against the pure-Go oracle on random inputs; CI runs it in the
// asmkernel job, and a kernel that fails it is not merged, no matter
// what its benchmark says.

import (
	"math"
	"math/rand"
	"testing"
)

const tol = 1e-4 // documented tolerance: float summation order may differ

func randSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

func TestAccelMatVecMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, dims := range [][2]int{{4, 8}, {64, 64}, {128, 64}, {260, 64}} {
		m, n := dims[0], dims[1]
		w, x := randSlice(rng, m*n), randSlice(rng, n)
		ref := Pure{}.MatVec(w, m, n, x)
		got := Active().MatVec(w, m, n, x)
		for i := range ref {
			if math.Abs(float64(ref[i]-got[i])) > tol {
				t.Fatalf("%dx%d row %d: accel %v vs oracle %v — semantics violated",
					m, n, i, got[i], ref[i])
			}
		}
	}
}

func TestAccelMatVecQ8MatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, dims := range [][2]int{{8, 16}, {128, 64}, {64, 128}} {
		m, n := dims[0], dims[1]
		w, x := randSlice(rng, m*n), randSlice(rng, n)
		q, s := Quantize8(w)
		ref := Pure{}.MatVecQ8(q, s, m, n, x)
		got := Active().MatVecQ8(q, s, m, n, x)
		for i := range ref {
			if math.Abs(float64(ref[i]-got[i])) > tol {
				t.Fatalf("%dx%d row %d: accel %v vs oracle %v — semantics violated",
					m, n, i, got[i], ref[i])
			}
		}
	}
}
