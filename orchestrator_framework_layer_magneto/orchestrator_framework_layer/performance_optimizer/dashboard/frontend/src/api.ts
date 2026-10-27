export interface StatBucket {
  n_calls: number;
  success_rate: number | null;
  avg_latency_ms: number | null;
  fallback_rate?: number | null;
  by_provider?: Record<string, { n_calls: number; success_rate: number; avg_latency_ms: number }>;
  by_arm?: Record<string, { n_calls: number; success_rate: number; avg_latency_ms: number }>;
}

export interface KernelTier {
  id: string;
  build: string;
  requires: string;
  mechanism: string;
}

export interface EngineStats {
  configured: boolean;
  reachable: boolean;
  error?: string;
  stats: {
    model: string;
    kernel_tier: string;
    resident_experts: number;
    lru_evictions: number;
    ram_hits: number;
    disk_misses: number;
    hit_rate: number;
  } | null;
}

export interface Summary {
  date: string;
  today: StatBucket;
  recent_dates: string[];
  provider_rl_posteriors: Record<string, Record<string, number>> | null;
  gpu: { kernel_tiers: KernelTier[]; live: EngineStats };
}

export interface DailyReport {
  date: string;
  today: StatBucket;
  previous_date: string;
  previous: StatBucket;
  delta: {
    success_rate: number | null;
    avg_latency_ms: number | null;
    fallback_rate: number | null;
    n_calls: number | null;
  };
}

export interface OptEvent {
  node: string;
  tier: string;
  provider: string;
  arm_id: string;
  ok: boolean;
  latency_ms: number;
  fallback_used: boolean;
  occurred_at: string;
}

const TOKEN_KEY = "opt_dashboard_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

async function authedFetch(path: string): Promise<Response> {
  const token = getToken();
  const res = await fetch(path, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (res.status === 401) {
    clearToken();
  }
  return res;
}

export async function login(username: string, password: string): Promise<string> {
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error("bad credentials");
  const body = await res.json();
  setToken(body.access_token);
  return body.access_token;
}

export async function fetchSummary(): Promise<Summary> {
  const res = await authedFetch("/api/summary");
  if (!res.ok) throw new Error("failed to load summary");
  return res.json();
}

export async function fetchDaily(date?: string): Promise<DailyReport> {
  const qs = date ? `?date=${date}` : "";
  const res = await authedFetch(`/api/daily${qs}`);
  if (!res.ok) throw new Error("failed to load daily report");
  return res.json();
}

export async function fetchEvents(date?: string): Promise<{ date: string; events: OptEvent[] }> {
  const qs = date ? `?date=${date}` : "";
  const res = await authedFetch(`/api/events${qs}`);
  if (!res.ok) throw new Error("failed to load events");
  return res.json();
}
