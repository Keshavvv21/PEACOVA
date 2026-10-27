package api

import (
	"embed"
	"io/fs"
	"net/http"
)

// The dashboard ships inside the binary: embed.FS compiles web/ into the
// executable, so `algo-engine web` needs no assets on disk — the single
// static binary promise extends to the UI. ('s dashboard is a
// React/Vite app served by `coli web` from web/dist; this vanilla
// single-file version is its toy-scale equivalent, and swapping in a
// built SPA later is just replacing the files under web/.)

//go:embed web
var webFS embed.FS

// dashboardHandler serves the embedded UI at /.
func dashboardHandler() http.Handler {
	sub, err := fs.Sub(webFS, "web")
	if err != nil {
		panic(err) // impossible: "web" is embedded above
	}
	return http.FileServer(http.FS(sub))
}
