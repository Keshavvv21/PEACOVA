// Package safetensors reads and writes the safetensors container format:
//
//	[8 bytes little-endian: header length N][N bytes JSON header][raw tensor data]
//
// The JSON header maps each tensor name to {dtype, shape, data_offsets},
// where data_offsets are [begin, end) byte ranges into the data section.
// The simplicity is the point: with the index in hand, any tensor is one
// ReadAt away — which is exactly what tiered streaming needs.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// TensorInfo describes one tensor's location inside the file.
type TensorInfo struct {
	DType       string  `json:"dtype"`        // "F32" or "I8" in this engine
	Shape       []int64 `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"` // [begin, end) within data section
}

// File is an open safetensors file with its parsed index.
// Reads use ReadAt (pread), so File is safe for concurrent readers.
type File struct {
	f        *os.File
	Index    map[string]TensorInfo
	dataBase int64 // file offset where the data section starts
}

// Open parses the header and returns a ready-to-read File.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	var lenBuf [8]byte
	if _, err := f.ReadAt(lenBuf[:], 0); err != nil {
		f.Close()
		return nil, fmt.Errorf("read header length: %w", err)
	}
	headerLen := int64(binary.LittleEndian.Uint64(lenBuf[:]))
	if headerLen <= 0 || headerLen > 100<<20 {
		f.Close()
		return nil, fmt.Errorf("implausible header length %d", headerLen)
	}
	headerJSON := make([]byte, headerLen)
	if _, err := f.ReadAt(headerJSON, 8); err != nil {
		f.Close()
		return nil, fmt.Errorf("read header: %w", err)
	}
	index := map[string]TensorInfo{}
	if err := json.Unmarshal(headerJSON, &index); err != nil {
		f.Close()
		return nil, fmt.Errorf("parse header json: %w", err)
	}
	delete(index, "__metadata__") // optional metadata block is not a tensor
	return &File{f: f, Index: index, dataBase: 8 + headerLen}, nil
}

func (s *File) Close() error { return s.f.Close() }

// ReadF32 reads a float32 tensor fully into memory.
func (s *File) ReadF32(name string) ([]float32, error) {
	info, ok := s.Index[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not in file", name)
	}
	if info.DType != "F32" {
		return nil, fmt.Errorf("tensor %q is %s, want F32", name, info.DType)
	}
	raw, err := s.readRange(info)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// ReadRaw reads a tensor's raw bytes (used for I8 quantized tensors).
func (s *File) ReadRaw(name string) ([]byte, TensorInfo, error) {
	info, ok := s.Index[name]
	if !ok {
		return nil, TensorInfo{}, fmt.Errorf("tensor %q not in file", name)
	}
	raw, err := s.readRange(info)
	return raw, info, err
}

// ReadRawAt reads length bytes of a tensor starting at byte offset within
// the tensor — the primitive the disk tier uses to stream one expert.
func (s *File) ReadRawAt(name string, offset, length int64, dst []byte) error {
	info, ok := s.Index[name]
	if !ok {
		return fmt.Errorf("tensor %q not in file", name)
	}
	if offset+length > info.DataOffsets[1]-info.DataOffsets[0] {
		return fmt.Errorf("read past end of tensor %q", name)
	}
	_, err := s.f.ReadAt(dst[:length], s.dataBase+info.DataOffsets[0]+offset)
	return err
}

func (s *File) readRange(info TensorInfo) ([]byte, error) {
	n := info.DataOffsets[1] - info.DataOffsets[0]
	buf := make([]byte, n)
	if _, err := s.f.ReadAt(buf, s.dataBase+info.DataOffsets[0]); err != nil {
		return nil, fmt.Errorf("read tensor bytes: %w", err)
	}
	return buf, nil
}

// ── Writing ──────────────────────────────────────────────────────────────

// Writer accumulates tensors and writes a valid safetensors file.
type Writer struct {
	names   []string
	infos   map[string]TensorInfo
	blobs   map[string][]byte
	cursor  int64
}

func NewWriter() *Writer {
	return &Writer{infos: map[string]TensorInfo{}, blobs: map[string][]byte{}}
}

// AddF32 registers a float32 tensor.
func (w *Writer) AddF32(name string, shape []int64, data []float32) {
	raw := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	w.add(name, "F32", shape, raw)
}

// AddI8 registers an int8 tensor from raw bytes.
func (w *Writer) AddI8(name string, shape []int64, raw []byte) {
	w.add(name, "I8", shape, raw)
}

func (w *Writer) add(name, dtype string, shape []int64, raw []byte) {
	w.names = append(w.names, name)
	w.infos[name] = TensorInfo{
		DType: dtype, Shape: shape,
		DataOffsets: [2]int64{w.cursor, w.cursor + int64(len(raw))},
	}
	w.blobs[name] = raw
	w.cursor += int64(len(raw))
}

// WriteFile emits header + data. Tensor order in the data section follows
// insertion order; the JSON header carries the offsets either way.
func (w *Writer) WriteFile(path string) error {
	header, err := json.Marshal(w.infos)
	if err != nil {
		return fmt.Errorf("marshal header: %w", err)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(header)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(header); err != nil {
		return err
	}
	// data section must follow the offsets recorded in infos: sort names
	// by begin offset to be explicit rather than relying on map order.
	names := append([]string(nil), w.names...)
	sort.Slice(names, func(i, j int) bool {
		return w.infos[names[i]].DataOffsets[0] < w.infos[names[j]].DataOffsets[0]
	})
	for _, n := range names {
		if _, err := f.Write(w.blobs[n]); err != nil {
			return err
		}
	}
	return nil
}
