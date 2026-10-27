package tokenizer

import "testing"

func TestByteRoundTrip(t *testing.T) {
	tok := NewByteLevel(nil)
	text := "hello, Algo Engine! 123"
	ids := tok.Encode(text)
	if got := tok.Decode(ids); got != text {
		t.Fatalf("round trip: got %q want %q", got, text)
	}
	if len(ids) != len(text) {
		t.Fatalf("byte-level should be 1 id per byte: %d vs %d", len(ids), len(text))
	}
}

func TestMergesReduceLength(t *testing.T) {
	merges := map[[2]string]int{
		{"t", "a"}:   0,
		{"ta", "s"}:  1,
		{"tas", "k"}: 2,
	}
	tok := NewByteLevel(merges)
	ids := tok.Encode("task")
	if len(ids) != 1 {
		t.Fatalf("want 1 merged token, got %d: %v", len(ids), ids)
	}
	if tok.Decode(ids) != "task" {
		t.Fatalf("decode: got %q", tok.Decode(ids))
	}
}

func TestMergePriorityOrder(t *testing.T) {
	// "ab" (rank 0) must merge before "bc" (rank 1) in "abc":
	// abc -> [ab, c]  (no bc pair survives)
	merges := map[[2]string]int{{"a", "b"}: 0, {"b", "c"}: 1}
	tok := NewByteLevel(merges)
	ids := tok.Encode("abc")
	if len(ids) != 2 {
		t.Fatalf("want [ab, c] = 2 tokens, got %d", len(ids))
	}
}
