package attention

import (
	"math"
	"testing"

	"algo-engine/internal/kvcache"
)

// identity returns an n×n identity matrix, row-major.
func identity(n int) []float32 {
	m := make([]float32, n*n)
	for i := 0; i < n; i++ {
		m[i*n+i] = 1
	}
	return m
}

func TestSingleTokenAttendsToItself(t *testing.T) {
	// With identity projections and one cached position, attention weight
	// is 1 on that position, so out == v == h.
	d := 4
	l := NewLayer(identity(d), identity(d), identity(d), identity(d), d)
	c := kvcache.New("s", 1)

	h := []float32{0.5, -1, 2, 0}
	out := l.Forward(h, c, 0)
	for i := range h {
		if math.Abs(float64(out[i]-h[i])) > 1e-6 {
			t.Fatalf("out=%v want %v", out, h)
		}
	}
	if c.Len() != 1 {
		t.Fatalf("cache len=%d want 1", c.Len())
	}
}

func TestCacheGrowsAndMixesHistory(t *testing.T) {
	d := 2
	// Wq = 0 makes every query the zero vector, so all scores are equal
	// and softmax weights are uniform -- history mixes 50/50.
	l := NewLayer(make([]float32, d*d), identity(d), identity(d), identity(d), d)
	c := kvcache.New("s", 1)

	l.Forward([]float32{1, 0}, c, 0)
	out := l.Forward([]float32{0, 1}, c, 0)

	if c.Len() != 2 {
		t.Fatalf("cache len=%d want 2", c.Len())
	}
	// uniform weights -> out = 0.5*[1,0] + 0.5*[0,1] = [0.5, 0.5]
	for i := range out {
		if math.Abs(float64(out[i]-0.5)) > 1e-6 {
			t.Fatalf("out=%v want [0.5 0.5]", out)
		}
	}
}
