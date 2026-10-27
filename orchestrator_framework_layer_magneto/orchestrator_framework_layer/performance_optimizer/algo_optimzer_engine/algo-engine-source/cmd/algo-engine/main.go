// Command algo-engine is the single binary: gen / chat / serve / doctor.
// (Production note: the design doc specifies cobra for the CLI; this
// stdlib version keeps the manual dependency-free — swapping to cobra is
// a mechanical change to this one file.)
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"

	"algo-engine/internal/api"
	"algo-engine/internal/engine"
	"algo-engine/internal/kvcache"
	"algo-engine/models"
	_ "algo-engine/models/olmoe" // register the olmoe family
	_ "algo-engine/models/toy"   // register the toy family
)

const version = "0.1.0"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	cmd, args := os.Args[1], os.Args[2:]
	var err error
	switch cmd {
	case "gen":
		err = runGen(args)
	case "chat":
		err = runChat(args)
	case "serve":
		err = runServe(args)
	case "web":
		err = runWeb(args)
	case "doctor":
		err = runDoctor(args)
	case "families":
		err = runFamilies(args)
	case "plan":
		err = runPlan(args)
	case "version":
		fmt.Println("algo-engine", version)
	default:
		usage()
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Println(`algo-engine — a -inspired local MoE inference engine in Go

  algo-engine gen    --model DIR [--seed N]        generate the toy model
  algo-engine chat   --model DIR [--session NAME]  interactive terminal chat
  algo-engine serve  --model DIR [--addr :8080]    OpenAI-compatible API
  algo-engine web    --model DIR [--addr :8080]    serve + open the dashboard
  algo-engine doctor --model DIR                   readiness check
  algo-engine families                             list registered model families
  algo-engine plan   --model DIR                   describe/plan a model directory
  algo-engine version`)
}

// printable replaces non-printable bytes with '·' for terminal display.
// The untrained toy model emits arbitrary bytes; the token stream is real,
// only its rendering is sanitized here.
func printable(s string) string {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b >= 32 && b < 127 || b == '\n' {
			out = append(out, b)
		} else {
			out = append(out, '.')
		}
	}
	return string(out)
}

func ctrlCContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(),
		os.Interrupt, syscall.SIGTERM)
}

func loadEngine(modelDir string) (*engine.Engine, error) {
	return engine.Load(modelDir, engine.Options{
		CacheCapacity:      32, // half the toy's 64 experts: tiering visible
		MaxConcurrentReads: 4,
		PinHottest:         8,
	})
}

// ── gen ──────────────────────────────────────────────────────────────────

func runGen(args []string) error {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	dir := fs.String("model", "toy-model", "output model directory")
	seed := fs.Int64("seed", 42, "weight seed (same seed = same model)")
	fs.Parse(args)

	if err := engine.GenerateToyModel(*dir, *seed); err != nil {
		return err
	}
	fmt.Printf("wrote toy model to %s (seed %d)\n", *dir, *seed)
	return nil
}

// ── chat ─────────────────────────────────────────────────────────────────

func runChat(args []string) error {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	dir := fs.String("model", "toy-model", "model directory")
	session := fs.String("session", "default", "session name (KV persistence)")
	maxNew := fs.Int("max-new", 48, "max tokens per reply")
	temp := fs.Float64("temp", 0.8, "sampling temperature (0 = greedy)")
	fs.Parse(args)

	ctx, cancel := ctrlCContext()
	defer cancel()

	start := time.Now()
	eng, err := loadEngine(*dir)
	if err != nil {
		return err
	}
	defer eng.Close(*dir)

	cache, err := kvcache.Load(*dir, *session, eng.Cfg.NLayers)
	if err != nil {
		return err
	}
	warm := ""
	if cache.Len() > 0 {
		warm = fmt.Sprintf(" · session warm (%d cached positions)", cache.Len())
	}
	fmt.Printf("🐦 algo-engine %s — %s · %d layers · %d experts/layer · int8\n",
		version, eng.Cfg.Name, eng.Cfg.NLayers, eng.Cfg.NExperts)
	fmt.Printf("✓ ready in %v · resident experts: %d%s\n",
		time.Since(start).Round(time.Millisecond), eng.Store.ResidentCount(), warm)
	fmt.Println("  (untrained toy weights: output is structured gibberish —")
	fmt.Println("   the pipeline is real, the intelligence is not. Ctrl-C or 'exit' to quit)")

	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n› ")
		if !sc.Scan() {
			break
		}
		line := sc.Text()
		if line == "exit" || line == "quit" {
			break
		}
		if ctx.Err() != nil {
			break
		}

		fmt.Print("◆ ")
		t0 := time.Now()
		ids, err := eng.Generate(ctx, line, *maxNew,
			engine.NewSampler(*temp, 40, time.Now().UnixNano()), cache,
			func(_ int, text string) { fmt.Print(printable(text)) })
		if err != nil && ctx.Err() == nil {
			return err
		}
		dt := time.Since(t0).Seconds()
		hits, misses := eng.Heat.Stats()
		hitRate := 0.0
		if total := hits + misses; total > 0 {
			hitRate = 100 * float64(hits) / float64(total)
		}
		fmt.Printf("\n  [%d tok · %.1f tok/s · hit-rate %.0f%% · resident %d]\n",
			len(ids), float64(len(ids))/dt, hitRate,
			eng.Store.ResidentCount())
		if ctx.Err() != nil {
			break
		}
	}

	if err := cache.Save(*dir); err != nil {
		return fmt.Errorf("save session: %w", err)
	}
	fmt.Println("\nsession saved — reopens warm.")
	return nil
}

// ── serve ────────────────────────────────────────────────────────────────

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	dir := fs.String("model", "toy-model", "model directory")
	addr := fs.String("addr", ":8080", "listen address")
	fs.Parse(args)

	ctx, cancel := ctrlCContext()
	defer cancel()

	eng, err := loadEngine(*dir)
	if err != nil {
		return err
	}
	defer eng.Close(*dir)

	return api.NewServer(eng, *dir).Run(ctx, *addr)
}

// ── web ──────────────────────────────────────────────────────────────────

// runWeb is serve plus a dashboard hint (and a best-effort browser open) —
// the coli web equivalent.
func runWeb(args []string) error {
	fs := flag.NewFlagSet("web", flag.ExitOnError)
	dir := fs.String("model", "toy-model", "model directory")
	addr := fs.String("addr", ":8080", "listen address")
	fs.Parse(args)

	url := "http://localhost" + *addr
	if (*addr)[0] != ':' {
		url = "http://" + *addr
	}
	fmt.Printf("dashboard: %s\n", url)
	go func() { // best-effort; headless machines just use the printed URL
		time.Sleep(300 * time.Millisecond)
		for _, opener := range [][]string{
			{"xdg-open", url}, {"open", url},
			{"rundll32", "url.dll,FileProtocolHandler", url},
		} {
			if exec.Command(opener[0], opener[1:]...).Start() == nil {
				return
			}
		}
	}()

	ctx, cancel := ctrlCContext()
	defer cancel()
	eng, err := loadEngine(*dir)
	if err != nil {
		return err
	}
	defer eng.Close(*dir)
	return api.NewServer(eng, *dir).Run(ctx, *addr)
}

// ── families / plan (the ModelEngine seam, user-visible) ─────────────────

func runFamilies(args []string) error {
	for _, name := range models.List() {
		fmt.Println(" ", name)
	}
	return nil
}

// runPlan auto-detects the family for a model directory and prints its
// description — for a toy dir, its shape; for an HF OLMoE checkpoint,
// the full conversion + memory plan ('s `coli plan`, in spirit).
func runPlan(args []string) error {
	fs := flag.NewFlagSet("plan", flag.ExitOnError)
	dir := fs.String("model", "toy-model", "model directory (ours or an HF checkpoint)")
	fs.Parse(args)

	m, err := models.Open(*dir)
	if err != nil {
		return err
	}
	desc, err := m.Describe(*dir)
	if err != nil {
		return err
	}
	fmt.Printf("family: %s\n%s\n", m.Family(), desc)
	return nil
}

// ── doctor ───────────────────────────────────────────────────────────────

func runDoctor(args []string) error {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	dir := fs.String("model", "toy-model", "model directory")
	fs.Parse(args)

	cfg, err := engine.LoadConfig(*dir)
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	fmt.Printf("✓ config.json         %s (%d layers × %d experts, top-%d)\n",
		cfg.Name, cfg.NLayers, cfg.NExperts, cfg.TopK)

	for _, f := range []string{"dense.safetensors", "experts.safetensors"} {
		st, err := os.Stat(*dir + "/" + f)
		if err != nil {
			return fmt.Errorf("✗ %s missing: %w", f, err)
		}
		fmt.Printf("✓ %-20s %d bytes\n", f, st.Size())
	}

	// deepest check: load and run one token end to end
	eng, err := loadEngine(*dir)
	if err != nil {
		return fmt.Errorf("engine load: %w", err)
	}
	defer eng.Close(*dir)
	cache := kvcache.New("doctor", cfg.NLayers)
	if _, err := eng.ForwardToken(context.Background(), 65, 0, cache); err != nil {
		return fmt.Errorf("forward pass: %w", err)
	}
	fmt.Println("✓ forward pass        one token routed, streamed, computed")
	fmt.Println("all checks passed.")
	return nil
}
