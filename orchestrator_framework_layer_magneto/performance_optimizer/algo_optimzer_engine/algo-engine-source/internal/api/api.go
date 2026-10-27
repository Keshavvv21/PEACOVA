// Package api exposes the engine over an OpenAI-compatible HTTP API —
// pure net/http, no Python gateway. Anything already written against an
// OpenAI-style endpoint can point at Algo Engine by changing one base
// URL: POST /v1/chat/completions works in both normal and streaming
// (Server-Sent Events) modes, and GET /stats reports tiering health.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"algo-engine/internal/engine"
	"algo-engine/internal/kvcache"
)

// Server serves one loaded engine. Requests are handled one at a time
// (the engine mutex is the KV cache + store; a slot system per session
// is the documented extension).
type Server struct {
	Eng      *engine.Engine
	ModelDir string
	sem      chan struct{} // serialize generations
}

func NewServer(eng *engine.Engine, modelDir string) *Server {
	return &Server{Eng: eng, ModelDir: modelDir, sem: make(chan struct{}, 1)}
}

// ── OpenAI-compatible request/response shapes (the subset we serve) ─────

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
}

// streaming delta chunk
type deltaChoice struct {
	Index        int             `json:"index"`
	Delta        map[string]any  `json:"delta"`
	FinishReason *string         `json:"finish_reason"`
}
type chatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []deltaChoice `json:"choices"`
}

// ── Handlers ─────────────────────────────────────────────────────────────

func (s *Server) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("GET /", dashboardHandler()) // embedded web dashboard
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("GET /stats", s.handleStats)
	return mux
}

// Run serves until ctx is cancelled, then shuts down gracefully.
func (s *Server) Run(ctx context.Context, addr string) error {
	srv := &http.Server{Addr: addr, Handler: s.routes()}
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutCtx)
	}()
	log.Printf("api: OpenAI-compatible endpoint on %s", addr)
	err := srv.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

// promptFromMessages renders the toy chat template: role-tagged lines
// ending with the assistant cue. Real model families each carry their own
// template — that is what a ModelEngine's ChatTemplate() abstracts.
func promptFromMessages(msgs []chatMessage) string {
	var b strings.Builder
	for _, m := range msgs {
		fmt.Fprintf(&b, "%s: %s\n", m.Role, m.Content)
	}
	b.WriteString("assistant:")
	return b.String()
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"bad json"}`, http.StatusBadRequest)
		return
	}
	if req.MaxTokens <= 0 || req.MaxTokens > 512 {
		req.MaxTokens = 64
	}

	// one generation at a time; queue politely, honor client disconnect
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-r.Context().Done():
		return
	}

	prompt := promptFromMessages(req.Messages)
	sampler := engine.NewSampler(req.Temperature, 40, time.Now().UnixNano())
	cache := kvcache.New("http", s.Eng.Cfg.NLayers) // fresh per request
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	if req.Stream {
		s.streamChat(w, r, prompt, req, sampler, cache, id, created)
		return
	}

	ids, err := s.Eng.Generate(r.Context(), prompt, req.MaxTokens, sampler,
		cache, nil)
	if err != nil && r.Context().Err() == nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err), http.StatusInternalServerError)
		return
	}
	resp := chatResponse{
		ID: id, Object: "chat.completion", Created: created,
		Model: s.Eng.Cfg.Name,
		Choices: []chatChoice{{
			Message:      chatMessage{Role: "assistant", Content: s.Eng.Tok.Decode(ids)},
			FinishReason: "stop",
		}},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// streamChat emits SSE chunks: one delta per generated token, then the
// finish chunk and the [DONE] sentinel, flushing after every event.
func (s *Server) streamChat(w http.ResponseWriter, r *http.Request,
	prompt string, req chatRequest, sampler *engine.Sampler,
	cache *kvcache.Cache, id string, created int64) {

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")

	send := func(c chatChunk) {
		raw, _ := json.Marshal(c)
		fmt.Fprintf(w, "data: %s\n\n", raw)
		flusher.Flush()
	}

	_, err := s.Eng.Generate(r.Context(), prompt, req.MaxTokens, sampler, cache,
		func(_ int, text string) {
			send(chatChunk{ID: id, Object: "chat.completion.chunk",
				Created: created, Model: s.Eng.Cfg.Name,
				Choices: []deltaChoice{{Delta: map[string]any{"content": text}}}})
		})
	if err != nil && r.Context().Err() == nil {
		log.Println("stream generate:", err)
	}
	stop := "stop"
	send(chatChunk{ID: id, Object: "chat.completion.chunk", Created: created,
		Model:   s.Eng.Cfg.Name,
		Choices: []deltaChoice{{Delta: map[string]any{}, FinishReason: &stop}}})
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	hits, misses := s.Eng.Heat.Stats()
	hitRate := 0.0
	if hits+misses > 0 {
		hitRate = float64(hits) / float64(hits+misses)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"model":            s.Eng.Cfg.Name,
		"kernel_tier":      s.Eng.KernelTier(),
		"resident_experts": s.Eng.Store.ResidentCount(),
		"lru_evictions":    s.Eng.Store.Evicted(),
		"ram_hits":         hits,
		"disk_misses":      misses,
		"hit_rate":         hitRate,
	})
}
