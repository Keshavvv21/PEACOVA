package safetensors

import (
	"path/filepath"
	"testing"
)

func TestWriteReadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.safetensors")

	w := NewWriter()
	w.AddF32("embed", []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	w.AddI8("expert.0", []int64{4}, []byte{0x01, 0xFF, 0x7F, 0x80})
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	got, err := f.ReadF32("embed")
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("embed[%d]=%v want %v", i, got[i], want[i])
		}
	}

	raw, info, err := f.ReadRaw("expert.0")
	if err != nil {
		t.Fatal(err)
	}
	if info.DType != "I8" || len(raw) != 4 || raw[1] != 0xFF {
		t.Fatalf("raw read wrong: %v %v", info, raw)
	}

	// partial read: bytes [1,3) of the tensor
	dst := make([]byte, 2)
	if err := f.ReadRawAt("expert.0", 1, 2, dst); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 0xFF || dst[1] != 0x7F {
		t.Fatalf("ReadRawAt got %v", dst)
	}
}

func TestMissingTensor(t *testing.T) {
	path := filepath.Join(t.TempDir(), "t.safetensors")
	w := NewWriter()
	w.AddF32("x", []int64{1}, []float32{1})
	if err := w.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	f, _ := Open(path)
	defer f.Close()
	if _, err := f.ReadF32("nope"); err == nil {
		t.Fatal("want error for missing tensor")
	}
}
