//go:build !asmkernel && !cuda

package quant

// Active returns the kernel this build uses. The default build — no tags,
// CGO_ENABLED=0, fully static — always selects the pure-Go oracle. This
// file and kernel_accel.go are mutually exclusive at compile time; exactly
// one Active() exists in any build.
func Active() Kernel { return Pure{} }
