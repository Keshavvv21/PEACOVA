package quant

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantizeDequantizeError(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	w := make([]float32, 1000)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q, scale := Quantize8(w)
	back := Dequant8(q, scale)

	// max error of symmetric int8 is scale/2 per weight
	maxErr := float64(scale) / 2 * 1.0001
	for i := range w {
		if e := math.Abs(float64(w[i] - back[i])); e > maxErr {
			t.Fatalf("weight %d error %.6f > %.6f", i, e, maxErr)
		}
	}
}

func TestByteRoundTrip(t *testing.T) {
	q := []int8{-127, -1, 0, 1, 127}
	got := Int8FromBytes(BytesFromInt8(q))
	for i := range q {
		if got[i] != q[i] {
			t.Fatalf("byte round trip: %v -> %v", q, got)
		}
	}
}

func TestQ8MatVecMatchesDequantPath(t *testing.T) {
	// The quantized kernel must equal dequantize-then-multiply exactly
	// (same operations, different order of the scale multiply).
	rng := rand.New(rand.NewSource(2))
	m, n := 8, 16
	w := make([]float32, m*n)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q, scale := Quantize8(w)

	var k Pure
	y1 := k.MatVecQ8(q, scale, m, n, x)
	y2 := k.MatVec(Dequant8(q, scale), m, n, x)
	for i := range y1 {
		if math.Abs(float64(y1[i]-y2[i])) > 1e-4 {
			t.Fatalf("row %d: %v vs %v", i, y1[i], y2[i])
		}
	}
}
