//go:build cuda

// The GPU tier's kernel side — Phase 4 of the Technical Design (§9.3,
// §17), a build-tag-gated backend behind the same ComputeBackend seam as
// every other kernel tier (Pure/accelKernel) — nothing above this file
// needs to know CUDA exists. Selected only by:
//
//	make cuda        # nvcc-compiles internal/quant/cuda/kernels.cu, then
//	                 # CGO_ENABLED=1 go build -tags cuda ./cmd/algo-engine
//
// The default build never touches this file, never needs nvcc, and stays
// a static binary — the One Rule's trajectory: Go everywhere, C at the
// kernel seam, CUDA one tag further out on the same seam.
//
// Residency model (the part that makes this a *tier* and not a shuttle):
// the first MatVec/MatVecQ8 against a given weight tensor uploads it to
// the device ONCE and caches the device pointer; every later call against
// the same tensor reuses it. Per-call traffic is the O(d_model) activation
// vector only. This is the kernel-level miniature of the full Phase 4
// design, where the upload moves into the expert store (upload on
// insert/Pin, ae_cuda_free on evict, "place" gaining a TierVRAM answer).
package quant

/*
#cgo CFLAGS: -O3
#cgo LDFLAGS: -L${SRCDIR}/cuda -laekernels -lcudart -lstdc++

#include <stdlib.h>
#include <stdint.h>

int ae_cuda_init(void);
int ae_cuda_upload(const void *host, size_t n, void **dev);
int ae_cuda_free(void *dev);
int ae_cuda_matvec_f32(const void *w_dev, int m, int n,
                       const float *x_host, float *y_host);
int ae_cuda_matvec_q8(const void *q_dev, float scale, int m, int n,
                      const float *x_host, float *y_host);
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// Active returns the CUDA kernel (cuda builds only). Initialization is
// checked once; a machine without a usable device fails loudly at first
// use rather than silently computing garbage.
func Active() Kernel { return &cudaKernel{resident: map[uintptr]unsafe.Pointer{}} }

type cudaKernel struct {
	once     sync.Once
	initErr  error
	mu       sync.Mutex
	resident map[uintptr]unsafe.Pointer // host ptr -> device ptr (upload-once cache)
}

func (k *cudaKernel) init() {
	k.once.Do(func() {
		if C.ae_cuda_init() != 0 {
			k.initErr = fmt.Errorf("cuda: no usable device (build ran with -tags cuda)")
		}
	})
	if k.initErr != nil {
		panic(k.initErr) // a mis-provisioned GPU build must not limp along on CPU silently
	}
}

// deviceCopy returns the cached device pointer for this host buffer,
// uploading it on first sight. Weights in this engine are loaded once and
// never mutated, so identity of the backing array is a stable cache key.
func (k *cudaKernel) deviceCopy(host unsafe.Pointer, n int) unsafe.Pointer {
	key := uintptr(host)
	k.mu.Lock()
	defer k.mu.Unlock()
	if dev, ok := k.resident[key]; ok {
		return dev
	}
	var dev unsafe.Pointer
	if C.ae_cuda_upload(host, C.size_t(n), (*unsafe.Pointer)(unsafe.Pointer(&dev))) != 0 {
		panic("cuda: upload failed")
	}
	k.resident[key] = dev
	return dev
}

func (k *cudaKernel) MatVec(w []float32, m, n int, x []float32) []float32 {
	k.init()
	y := make([]float32, m)
	if m == 0 || n == 0 {
		return y
	}
	dev := k.deviceCopy(unsafe.Pointer(&w[0]), len(w)*4)
	if C.ae_cuda_matvec_f32(dev, C.int(m), C.int(n),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0]))) != 0 {
		panic("cuda: matvec_f32 failed")
	}
	return y
}

func (k *cudaKernel) Name() string { return "cuda" }

func (k *cudaKernel) MatVecQ8(q []int8, scale float32, m, n int, x []float32) []float32 {
	k.init()
	y := make([]float32, m)
	if m == 0 || n == 0 {
		return y
	}
	dev := k.deviceCopy(unsafe.Pointer(&q[0]), len(q))
	if C.ae_cuda_matvec_q8(dev, C.float(scale), C.int(m), C.int(n),
		(*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&y[0]))) != 0 {
		panic("cuda: matvec_q8 failed")
	}
	return y
}
