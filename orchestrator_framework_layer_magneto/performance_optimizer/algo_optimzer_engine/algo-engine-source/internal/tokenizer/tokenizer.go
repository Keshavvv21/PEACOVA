// Package tokenizer implements a byte-level BPE tokenizer: every byte is a
// base token, and a learned merge table combines frequent adjacent pairs
// into larger tokens. With an empty merge table it degrades gracefully to
// plain byte-level tokenization, which is what the toy model uses.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Special token IDs sit immediately after the 256 byte tokens.
const (
	ByteVocab = 256
	BOS       = 256 // beginning of sequence
	EOS       = 257 // end of sequence
	PAD       = 258
	UNK       = 259
	VocabSize = 260
)

// Tokenizer holds the vocabulary and the BPE merge table.
type Tokenizer struct {
	vocab  map[string]int    // token string -> id
	ids    []string          // id -> token string
	merges map[[2]string]int // pair -> merge priority (lower merges first)
}

// NewByteLevel builds the base tokenizer: 256 byte tokens + specials,
// with an optional merge table on top.
func NewByteLevel(merges map[[2]string]int) *Tokenizer {
	t := &Tokenizer{
		vocab:  make(map[string]int, VocabSize),
		ids:    make([]string, 0, VocabSize),
		merges: merges,
	}
	for b := 0; b < ByteVocab; b++ {
		s := string([]byte{byte(b)})
		t.vocab[s] = b
		t.ids = append(t.ids, s)
	}
	for _, sp := range []struct {
		id   int
		name string
	}{{BOS, "<bos>"}, {EOS, "<eos>"}, {PAD, "<pad>"}, {UNK, "<unk>"}} {
		t.vocab[sp.name] = sp.id
		t.ids = append(t.ids, sp.name)
	}
	// register merged tokens so Encode can look them up
	for pair := range merges {
		merged := pair[0] + pair[1]
		if _, ok := t.vocab[merged]; !ok {
			t.vocab[merged] = len(t.ids)
			t.ids = append(t.ids, merged)
		}
	}
	return t
}

// Encode turns text into token IDs by starting from raw bytes and applying
// BPE merges in priority order until no known pair remains.
func (t *Tokenizer) Encode(text string) []int {
	symbols := make([]string, 0, len(text))
	for i := 0; i < len(text); i++ { // bytes, not runes: byte-level BPE
		symbols = append(symbols, string(text[i]))
	}
	symbols = t.applyMerges(symbols)

	ids := make([]int, 0, len(symbols))
	for _, s := range symbols {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, UNK)
		}
	}
	return ids
}

// applyMerges repeatedly merges the adjacent pair with the lowest priority.
func (t *Tokenizer) applyMerges(symbols []string) []string {
	if len(t.merges) == 0 {
		return symbols
	}
	for {
		bestRank := int(^uint(0) >> 1) // max int
		bestPair := [2]string{}
		found := false
		for i := 0; i < len(symbols)-1; i++ {
			pair := [2]string{symbols[i], symbols[i+1]}
			if rank, ok := t.merges[pair]; ok && rank < bestRank {
				bestPair, bestRank, found = pair, rank, true
			}
		}
		if !found {
			return symbols
		}
		symbols = mergeOnce(symbols, bestPair)
	}
}

func mergeOnce(symbols []string, pair [2]string) []string {
	out := make([]string, 0, len(symbols))
	for i := 0; i < len(symbols); {
		if i < len(symbols)-1 && symbols[i] == pair[0] && symbols[i+1] == pair[1] {
			out = append(out, pair[0]+pair[1])
			i += 2
		} else {
			out = append(out, symbols[i])
			i++
		}
	}
	return out
}

// Decode maps IDs back to text. Special tokens are skipped; merged tokens
// decode to their full string.
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.ids) {
			continue
		}
		if id >= ByteVocab && id < VocabSize {
			continue // skip special tokens
		}
		sb.WriteString(t.ids[id])
	}
	return sb.String()
}

// tokenizerJSON mirrors the on-disk format: {"merges": ["a b", "ab c", ...]}.
// Merge priority is the entry's position in the list, GPT-2 style.
type tokenizerJSON struct {
	Merges []string `json:"merges"`
}

// LoadMerges reads a tokenizer.json-style merge list from disk.
func LoadMerges(path string) (map[[2]string]int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer file: %w", err)
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer json: %w", err)
	}
	merges := make(map[[2]string]int, len(tj.Merges))
	for i, line := range tj.Merges {
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("merge line %d malformed: %q", i, line)
		}
		merges[[2]string{parts[0], parts[1]}] = i
	}
	return merges, nil
}
