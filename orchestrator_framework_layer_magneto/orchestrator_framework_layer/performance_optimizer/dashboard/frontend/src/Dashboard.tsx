import { useEffect, useState } from "react";
import {
  DailyReport,
  OptEvent,
  Summary,
  clearToken,
  fetchDaily,
  fetchEvents,
  fetchSummary,
} from "./api";

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat-card">
      <div className="stat-card-label">{label}</div>
      <div className="stat-card-value">{value}</div>
      {sub && <div className="stat-card-sub">{sub}</div>}
    </div>
  );
}

function Delta({ value, invert = false }: { value: number | null; invert?: boolean }) {
  if (value === null) return <span className="delta neutral">no prior data</span>;
  const good = invert ? value < 0 : value > 0;
  const flat = Math.abs(value) < 1e-9;
  const cls = flat ? "neutral" : good ? "good" : "bad";
  const arrow = flat ? "→" : value > 0 ? "▲" : "▼";
  return (
    <span className={`delta ${cls}`}>
      {arrow} {Math.abs(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}
    </span>
  );
}

function Bar({ pct, tone = "ok" }: { pct: number; tone?: "ok" | "warn" }) {
  return (
    <div className="bar-track">
      <div className={`bar-fill ${tone}`} style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
    </div>
  );
}

function pct(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : `${Math.round(x * 100)}%`;
}
function ms(x: number | null | undefined): string {
  return x === null || x === undefined ? "—" : `${Math.round(x)} ms`;
}

export default function Dashboard({ onLogout }: { onLogout: () => void }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [daily, setDaily] = useState<DailyReport | null>(null);
  const [events, setEvents] = useState<OptEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);

  async function loadAll() {
    try {
      const [s, d, e] = await Promise.all([fetchSummary(), fetchDaily(), fetchEvents()]);
      setSummary(s);
      setDaily(d);
      setEvents(e.events.slice(-25).reverse());
      setErr(null);
    } catch {
      setErr("Session expired or the API is unreachable.");
    }
  }

  useEffect(() => {
    loadAll();
    const id = setInterval(loadAll, 15000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function logout() {
    clearToken();
    onLogout();
  }

  if (err) {
    return (
      <div className="dash-shell">
        <div className="error-banner">{err}</div>
        <button onClick={logout}>Back to login</button>
      </div>
    );
  }
  if (!summary || !daily) {
    return <div className="dash-shell loading">Loading…</div>;
  }

  const t = summary.today;
  const live = summary.gpu.live;

  return (
    <div className="dash-shell">
      <header className="dash-header">
        <div>
          <h1>Optimization Dashboard</h1>
          <p className="dash-sub">{summary.date} · auto-refreshes every 15s</p>
        </div>
        <button className="logout-btn" onClick={logout}>
          Sign out
        </button>
      </header>

      <section className="stat-grid">
        <StatCard label="Calls today" value={String(t.n_calls)} />
        <StatCard label="Success rate" value={pct(t.success_rate)} />
        <StatCard label="Avg latency" value={ms(t.avg_latency_ms)} />
        <StatCard label="Fallback rate" value={pct(t.fallback_rate)} sub="chain reorders that still failed rung 1" />
      </section>

      <section className="panel">
        <h2>Daily feedback loop — vs. {daily.previous_date}</h2>
        <div className="feedback-grid">
          <div>
            <div className="feedback-label">Success rate</div>
            <Delta value={daily.delta.success_rate} />
          </div>
          <div>
            <div className="feedback-label">Avg latency</div>
            <Delta value={daily.delta.avg_latency_ms} invert />
          </div>
          <div>
            <div className="feedback-label">Fallback rate</div>
            <Delta value={daily.delta.fallback_rate} invert />
          </div>
          <div>
            <div className="feedback-label">Call volume</div>
            <Delta value={daily.delta.n_calls} />
          </div>
        </div>
        <p className="panel-note">
          A rising success rate and falling latency day-over-day is optimization_router's and
          provider_optimizer's Thompson Sampling bandits actually learning — not a self-reported claim.
        </p>
      </section>

      <section className="panel">
        <h2>GPU / kernel tier</h2>
        <div className="kernel-live">
          {live.configured ? (
            live.reachable && live.stats ? (
              <div className="kernel-live-ok">
                <span className="kernel-badge">{live.stats.kernel_tier}</span>
                <span>
                  {live.stats.model} · hit rate {pct(live.stats.hit_rate)} · {live.stats.resident_experts}{" "}
                  resident experts
                </span>
              </div>
            ) : (
              <div className="kernel-live-down">Algo Engine configured but unreachable right now.</div>
            )
          ) : (
            <div className="kernel-live-down">
              ALGO_ENGINE_BASE_URL not configured on this dashboard — showing tier capabilities only.
            </div>
          )}
        </div>
        <div className="kernel-grid">
          {summary.gpu.kernel_tiers.map((k) => {
            const isLive = live.stats?.kernel_tier === k.id;
            return (
              <div key={k.id} className={`kernel-card ${isLive ? "kernel-active" : ""}`}>
                <div className="kernel-card-head">
                  <span className="kernel-id">{k.id}</span>
                  {isLive && <span className="kernel-live-tag">ACTIVE</span>}
                </div>
                <div className="kernel-build">{k.build}</div>
                <div className="kernel-requires">{k.requires}</div>
                <p className="kernel-mechanism">{k.mechanism}</p>
              </div>
            );
          })}
        </div>
      </section>

      <section className="panel">
        <h2>RL provider ordering — provider_optimizer posteriors</h2>
        {summary.provider_rl_posteriors ? (
          Object.entries(summary.provider_rl_posteriors).map(([tier, providers]) => {
            const max = Math.max(...Object.values(providers), 1e-9);
            return (
              <div key={tier} className="tier-block">
                <div className="tier-name">{tier}</div>
                {Object.entries(providers).map(([provider, mean]) => (
                  <div key={provider} className="rl-row">
                    <span className="rl-provider">{provider}</span>
                    <Bar pct={(mean / max) * 100} />
                    <span className="rl-mean">{mean.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            );
          })
        ) : (
          <p className="panel-note">
            No persisted posterior yet — provider_optimizer saves every 20 calls; keep chatting through
            Algo Akanade and this fills in.
          </p>
        )}
      </section>

      <section className="panel">
        <h2>By provider — today</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Provider</th>
              <th>Calls</th>
              <th>Success</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(t.by_provider ?? {}).map(([provider, v]) => (
              <tr key={provider}>
                <td>{provider}</td>
                <td>{v.n_calls}</td>
                <td>{pct(v.success_rate)}</td>
                <td>{ms(v.avg_latency_ms)}</td>
              </tr>
            ))}
            {Object.keys(t.by_provider ?? {}).length === 0 && (
              <tr>
                <td colSpan={4} className="empty-row">
                  No calls yet today.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>By optimization arm — today</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Arm</th>
              <th>Calls</th>
              <th>Success</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(t.by_arm ?? {}).map(([arm, v]) => (
              <tr key={arm}>
                <td>{arm}</td>
                <td>{v.n_calls}</td>
                <td>{pct(v.success_rate)}</td>
                <td>{ms(v.avg_latency_ms)}</td>
              </tr>
            ))}
            {Object.keys(t.by_arm ?? {}).length === 0 && (
              <tr>
                <td colSpan={4} className="empty-row">
                  No calls yet today.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="panel">
        <h2>Recent events</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Node</th>
              <th>Tier</th>
              <th>Provider</th>
              <th>Arm</th>
              <th>OK</th>
              <th>Latency</th>
              <th>Fallback</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i}>
                <td>{e.occurred_at.slice(11, 19)}</td>
                <td>{e.node}</td>
                <td>{e.tier}</td>
                <td>{e.provider}</td>
                <td>{e.arm_id}</td>
                <td className={e.ok ? "ok-yes" : "ok-no"}>{e.ok ? "✓" : "✗"}</td>
                <td>{ms(e.latency_ms)}</td>
                <td>{e.fallback_used ? "yes" : ""}</td>
              </tr>
            ))}
            {events.length === 0 && (
              <tr>
                <td colSpan={8} className="empty-row">
                  No events yet today.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}
