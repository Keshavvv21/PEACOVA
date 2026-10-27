// kernels.cu — the CUDA half of Algo Engine's optional GPU tier.
//
// Build (never automatic; see the Makefile's `cuda` target):
//     nvcc -O3 -arch=native -c kernels.cu -o kernels.o
//     ar rcs libaekernels.a kernels.o
// then:  CGO_ENABLED=1 go build -tags cuda ./cmd/algo-engine
//
// Design, mirrored from 's backend_cuda.cu at toy scale:
//   - weights are UPLOADED ONCE and stay resident on the device
//     (ae_cuda_upload returns a device pointer the Go side caches) —
//     residency, not per-call shuttling, is what makes a VRAM tier a tier;
//   - the activation vector x and result y are small (d_model-sized) and
//     are copied per call — that traffic is O(d), the weights are O(d²);
//   - one thread block row-parallelizes the matvec: thread i owns output
//     row i. Naive on purpose: the correctness rung. 's kernels
//     (quant_matmul, w4a16_matmul, fused gate/up + silu_mul) are the
//     shape this grows into, each step gated by the equivalence test.
//
// Everything is extern "C" so cgo can link it with plain C linkage.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define AE_CHECK(call)                                                    \
    do {                                                                  \
        cudaError_t e_ = (call);                                          \
        if (e_ != cudaSuccess) {                                          \
            fprintf(stderr, "cuda: %s at %s:%d\n",                        \
                    cudaGetErrorString(e_), __FILE__, __LINE__);          \
            return -1;                                                    \
        }                                                                 \
    } while (0)

extern "C" {

// ae_cuda_init: -1 if no usable device; 0 otherwise.
int ae_cuda_init(void) {
    int n = 0;
    AE_CHECK(cudaGetDeviceCount(&n));
    if (n < 1) { fprintf(stderr, "cuda: no devices\n"); return -1; }
    AE_CHECK(cudaSetDevice(0));
    return 0;
}

// ae_cuda_upload: copy n bytes of host weights to a fresh device buffer.
// Returns the device pointer via *dev; the caller (the Go kernel's
// residency cache — and later, the expert store's VRAM tier) owns it.
int ae_cuda_upload(const void *host, size_t n, void **dev) {
    AE_CHECK(cudaMalloc(dev, n));
    AE_CHECK(cudaMemcpy(*dev, host, n, cudaMemcpyHostToDevice));
    return 0;
}

int ae_cuda_free(void *dev) {
    AE_CHECK(cudaFree(dev));
    return 0;
}

// ── kernels ──────────────────────────────────────────────────────────────

__global__ static void matvec_f32_kernel(const float *w, int m, int n,
                                         const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const float *row = w + (long)i * n;
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc += row[j] * x[j];
    y[i] = acc;
}

__global__ static void matvec_q8_kernel(const int8_t *q, float scale,
                                        int m, int n,
                                        const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const int8_t *row = q + (long)i * n;
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc += (float)row[j] * x[j];
    y[i] = acc * scale;
}

// ── host-callable wrappers (weights already device-resident) ─────────────

int ae_cuda_matvec_f32(const void *w_dev, int m, int n,
                       const float *x_host, float *y_host) {
    float *x_d = NULL, *y_d = NULL;
    AE_CHECK(cudaMalloc(&x_d, (size_t)n * sizeof(float)));
    AE_CHECK(cudaMalloc(&y_d, (size_t)m * sizeof(float)));
    AE_CHECK(cudaMemcpy(x_d, x_host, (size_t)n * sizeof(float),
                        cudaMemcpyHostToDevice));
    int threads = 128, blocks = (m + threads - 1) / threads;
    matvec_f32_kernel<<<blocks, threads>>>((const float *)w_dev, m, n, x_d, y_d);
    AE_CHECK(cudaGetLastError());
    AE_CHECK(cudaMemcpy(y_host, y_d, (size_t)m * sizeof(float),
                        cudaMemcpyDeviceToHost));
    cudaFree(x_d); cudaFree(y_d);
    return 0;
}

int ae_cuda_matvec_q8(const void *q_dev, float scale, int m, int n,
                      const float *x_host, float *y_host) {
    float *x_d = NULL, *y_d = NULL;
    AE_CHECK(cudaMalloc(&x_d, (size_t)n * sizeof(float)));
    AE_CHECK(cudaMalloc(&y_d, (size_t)m * sizeof(float)));
    AE_CHECK(cudaMemcpy(x_d, x_host, (size_t)n * sizeof(float),
                        cudaMemcpyHostToDevice));
    int threads = 128, blocks = (m + threads - 1) / threads;
    matvec_q8_kernel<<<blocks, threads>>>((const int8_t *)q_dev, scale,
                                          m, n, x_d, y_d);
    AE_CHECK(cudaGetLastError());
    AE_CHECK(cudaMemcpy(y_host, y_d, (size_t)m * sizeof(float),
                        cudaMemcpyDeviceToHost));
    cudaFree(x_d); cudaFree(y_d);
    return 0;
}

} // extern "C"
