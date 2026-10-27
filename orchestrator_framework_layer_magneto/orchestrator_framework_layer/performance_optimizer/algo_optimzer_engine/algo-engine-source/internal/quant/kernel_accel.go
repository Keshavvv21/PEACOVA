//go:build asmkernel && !cuda

// The tip of the triangle: the ONE place C is allowed in this codebase,
// selected only by `go build -tags asmkernel`, and policed by the
// equivalence test in equiv_test.go. If this file is deleted the engine
// still builds and runs — that is the point.
package quant

/*
#cgo CFLAGS: -O3 -march=native

#include <stdint.h>

// y = W·x for a row-major m×n float matrix.
// A plain loop on purpose: -O3 -march=native autovectorizes it (AVX2 on
// x86-64, NEON on arm64), which is the honest first rung of "accelerated".
// Hand-written intrinsics/assembly are the next rung, behind the same tag.
static void matvec_f32(const float* w, int m, int n,
                       const float* x, float* y) {
    for (int i = 0; i < m; i++) {
        const float* row = w + (long)i * n;
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += row[j] * x[j];
        y[i] = acc;
    }
}

// y[i] = scale * sum_j q[i*n+j] * x[j] — the quantized product computed
// directly on int8 weights, one scale multiply per row. The int8 dot
// product is exactly the shape VPMADDUBSW (AVX2) and SDOT (NEON) exist
// for; the widened accumulator mirrors the Go oracle's semantics.
static void matvec_q8(const int8_t* q, float scale, int m, int n,
                      const float* x, float* y) {
    for (int i = 0; i < m; i++) {
        const int8_t* row = q + (long)i * n;
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += (float)row[j] * x[j];
        y[i] = acc * scale;
    }
}
*/
import "C"
import "unsafe"

// accelKernel implements Kernel via the C loops above.
type accelKernel struct{}

// Active returns the cgo-accelerated kernel (asmkernel builds only).
func Active() Kernel { return accelKernel{} }

func (accelKernel) Name() string { return "cpu-accel" }

func (accelKernel) MatVec(w []float32, m, n int, x []float32) []float32 {
	y := make([]float32, m)
	if m == 0 || n == 0 {
		return y
	}
	C.matvec_f32((*C.float)(unsafe.Pointer(&w[0])), C.int(m), C.int(n),
		(*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
	return y
}

func (accelKernel) MatVecQ8(q []int8, scale float32, m, n int, x []float32) []float32 {
	y := make([]float32, m)
	if m == 0 || n == 0 {
		return y
	}
	C.matvec_q8((*C.int8_t)(unsafe.Pointer(&q[0])), C.float(scale),
		C.int(m), C.int(n), (*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0])))
	return y
}
