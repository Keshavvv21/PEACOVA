.PHONY: build test race vet run cuda accel

build:
	go build -o bin/algo-engine ./cmd/algo-engine

test:
	go test ./...

race:
	go test -race ./...

vet:
	go vet ./...

# Optional cgo kernel (Step 16): needs a C compiler.
accel:
	CGO_ENABLED=1 go build -tags asmkernel -o bin/algo-engine-accel ./cmd/algo-engine

# Optional GPU tier (Phase 4): needs the NVIDIA toolkit. Never automatic.
cuda:
	nvcc -O3 -arch=native -c internal/quant/cuda/kernels.cu -o internal/quant/cuda/kernels.o
	ar rcs internal/quant/cuda/libaekernels.a internal/quant/cuda/kernels.o
	CGO_ENABLED=1 go build -tags cuda -o bin/algo-engine-cuda ./cmd/algo-engine
