"""Per-user OAuth 2.0 connections (GitHub/Linear/Google) — see
app/oauth_store.py for the token store and app/routers/oauth.py for the
HTTP endpoints. This package holds the static provider registry
(providers.py) and the signed-state CSRF mechanism (state.py) the
authorize/callback pair relies on."""
