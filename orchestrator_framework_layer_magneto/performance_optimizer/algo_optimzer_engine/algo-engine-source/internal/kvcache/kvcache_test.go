package kvcache

import (
	"testing"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	c := New("sess-1", 2)
	c.Append(0, []float32{1, 2}, []float32{3, 4})
	c.Append(1, []float32{5, 6}, []float32{7, 8})
	if err := c.Save(dir); err != nil {
		t.Fatal(err)
	}

	got, err := Load(dir, "sess-1", 2)
	if err != nil {
		t.Fatal(err)
	}
	if got.Len() != 1 || got.Keys[1][0][1] != 6 {
		t.Fatalf("loaded cache wrong: %+v", got)
	}
}

func TestLoadMissingIsColdStart(t *testing.T) {
	c, err := Load(t.TempDir(), "new-session", 3)
	if err != nil || c.Len() != 0 || c.NLayers != 3 {
		t.Fatalf("cold start: %v %+v", err, c)
	}
}

func TestLoadLayerMismatchFails(t *testing.T) {
	dir := t.TempDir()
	c := New("s", 2)
	c.Save(dir)
	if _, err := Load(dir, "s", 4); err == nil {
		t.Fatal("layer mismatch must error")
	}
}
