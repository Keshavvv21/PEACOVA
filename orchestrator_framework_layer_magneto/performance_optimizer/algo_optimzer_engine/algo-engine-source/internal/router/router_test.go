package router

import (
	"math"
	"testing"
)

func TestRoutePicksHighestLogits(t *testing.T) {
	// 4 experts, dModel=2. Rows of W are expert score vectors.
	// h = [1,0] -> logits are the first column: [0.1, 5.0, -1.0, 3.0]
	w := []float32{
		0.1, 9, // expert 0
		5.0, 9, // expert 1  <- highest
		-1., 9, // expert 2
		3.0, 9, // expert 3  <- second
	}
	g := NewGate(w, 4, 2, 2)
	choices := g.Route([]float32{1, 0})

	if len(choices) != 2 || choices[0].ID != 1 || choices[1].ID != 3 {
		t.Fatalf("got %+v, want experts 1 then 3", choices)
	}
	sum := choices[0].Weight + choices[1].Weight
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("blend weights sum %v, want 1", sum)
	}
	if choices[0].Weight <= choices[1].Weight {
		t.Fatalf("higher logit must get higher weight: %+v", choices)
	}
}

func TestRouteDeterministicOnTies(t *testing.T) {
	w := make([]float32, 4*2) // all-zero logits: 4-way tie
	g := NewGate(w, 4, 2, 2)
	c := g.Route([]float32{1, 1})
	if c[0].ID != 0 || c[1].ID != 1 {
		t.Fatalf("tie break must prefer lower IDs, got %+v", c)
	}
}
