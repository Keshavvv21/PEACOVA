// Package quant implements int8 symmetric quantization and the compute
// kernels (dequantize, matrix-vector multiply) the engine's hot path uses.
//
// Symmetric per-tensor int8: each float w is stored as round(w/scale) in
// [-127, 127], with one float32 scale per tensor chosen so the largest
// |w| maps to 127. Dequantization is w ≈ int8 * scale. Storage shrinks 4×
// and, because scale only multiplies, the matmul can even run directly on
// int8 with one final scale — the optimization Step 15's kernel exploits.
package quant

import "math"

// Quantize8 converts float32 weights to int8 + a single scale.
// A zero tensor gets scale 1 to avoid division by zero.
func Quantize8(w []float32) (q []int8, scale float32) {
	maxAbs := float32(0)
	for _, v := range w {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		return make([]int8, len(w)), 1
	}
	scale = maxAbs / 127
	q = make([]int8, len(w))
	for i, v := range w {
		r := math.Round(float64(v / scale))
		if r > 127 {
			r = 127
		} else if r < -127 {
			r = -127
		}
		q[i] = int8(r)
	}
	return q, scale
}

// Dequant8 expands int8 weights back to float32.
func Dequant8(q []int8, scale float32) []float32 {
	out := make([]float32, len(q))
	for i, v := range q {
		out[i] = float32(v) * scale
	}
	return out
}

// Int8FromBytes reinterprets raw file bytes as int8 weights (two's
// complement, which is how Quantize8's output is written to disk).
func Int8FromBytes(raw []byte) []int8 {
	out := make([]int8, len(raw))
	for i, b := range raw {
		out[i] = int8(b)
	}
	return out
}

// BytesFromInt8 is the inverse, used by the weight converter.
func BytesFromInt8(q []int8) []byte {
	out := make([]byte, len(q))
	for i, v := range q {
		out[i] = byte(v)
	}
	return out
}

// ── Kernels ──────────────────────────────────────────────────────────────

// Kernel is the seam where an accelerated implementation may be plugged in
// (Step 15). The pure-Go implementation below is the correctness oracle:
// any accelerated kernel must match it within tolerance in CI.
type Kernel interface {
	// MatVec computes y = W·x for a row-major M×N matrix W and length-N x,
	// returning length-M y.
	MatVec(w []float32, m, n int, x []float32) []float32
	// MatVecQ8 computes y = (q·scale)·x without materializing the
	// dequantized matrix: the int8 dot product is accumulated in int32
	// and scaled once per row.
	MatVecQ8(q []int8, scale float32, m, n int, x []float32) []float32
	// Name reports which tier is actually active in THIS build — "pure-go",
	// "cpu-accel", or "cuda" — the one fact a build tag alone doesn't
	// expose to anything outside this package. The performance_optimizer
	// dashboard's GPU panel reads this via GET /stats (see internal/api)
	// rather than trying to infer it from build flags it can't see.
	Name() string
}

// Pure is the reference pure-Go kernel.
type Pure struct{}

func (Pure) Name() string { return "pure-go" }

func (Pure) MatVec(w []float32, m, n int, x []float32) []float32 {
	y := make([]float32, m)
	for i := 0; i < m; i++ {
		row := w[i*n : (i+1)*n]
		var sum float32
		for j, v := range row {
			sum += v * x[j]
		}
		y[i] = sum
	}
	return y
}

func (Pure) MatVecQ8(q []int8, scale float32, m, n int, x []float32) []float32 {
	y := make([]float32, m)
	for i := 0; i < m; i++ {
		row := q[i*n : (i+1)*n]
		var sum float32
		for j, v := range row {
			sum += float32(v) * x[j]
		}
		y[i] = sum * scale
	}
	return y
}
