import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Chart, Step } from "../api";
import type { Message } from "./Chat";

export function MessageView({ message, onAsk, busy }: { message: Message; onAsk?: (q: string) => void; busy?: boolean }) {
  const isUser = message.role === "user";
  return (
    <div className={`message ${message.role} ${message.error ? "error" : ""}`}>
      <Avatar user={isUser} />
      <div className="bubble">
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <>
            <div className="markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
            {message.charts.length > 0 && <ChartGallery charts={message.charts} />}
            {message.steps.length > 0 && <Trace steps={message.steps} mode={message.mode} elapsed={message.elapsed_ms} />}
            {!!message.suggestions?.length && onAsk && (
              <div className="followups">
                <span className="muted">Try next:</span>
                {message.suggestions.map((q) => (
                  <button key={q} className="chip small" onClick={() => onAsk(q)} disabled={busy}>
                    {q}
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function Avatar({ user }: { user: boolean }) {
  return (
    <div className={`avatar ${user ? "user" : "bot"}`} aria-hidden>
      {user ? (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="8" r="4" />
          <path d="M4 21c0-4 3.6-7 8-7s8 3 8 7" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="8" width="16" height="12" rx="3" />
          <path d="M12 8V4M8 4h8" />
          <circle cx="9" cy="14" r="1.2" fill="currentColor" stroke="none" />
          <circle cx="15" cy="14" r="1.2" fill="currentColor" stroke="none" />
        </svg>
      )}
    </div>
  );
}

function ChartGallery({ charts }: { charts: Chart[] }) {
  const [open, setOpen] = useState<Chart | null>(null);
  return (
    <>
      <div className={`charts ${charts.length > 1 ? "grid" : ""}`}>
        {charts.map((c) => (
          <figure key={c.id} className="chart-card" onClick={() => setOpen(c)}>
            <img src={c.url} alt={c.title} loading="lazy" />
            <figcaption>
              <span className={`chart-kind ${c.kind}`}>{c.kind}</span> {c.title}
            </figcaption>
          </figure>
        ))}
      </div>
      {open && (
        <div className="lightbox" onClick={() => setOpen(null)} role="dialog" aria-label={open.title}>
          <img src={open.url} alt={open.title} />
          <a className="ghost-btn" href={open.url} download onClick={(e) => e.stopPropagation()}>
            ⬇ Download PNG
          </a>
        </div>
      )}
    </>
  );
}

function Trace({ steps, mode, elapsed }: { steps: Step[]; mode?: string; elapsed?: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="trace">
      <button className="trace-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span className="trace-arrow">{open ? "▾" : "▸"}</span>
        {steps.length} tool call{steps.length === 1 ? "" : "s"}
        <span className="trace-tools">
          {steps.map((s, i) => (
            <code key={i}>{s.tool}</code>
          ))}
        </span>
        {elapsed !== undefined && <span className="muted">{(elapsed / 1000).toFixed(1)}s</span>}
        {mode && <span className={`mode-pill ${mode}`}>{mode === "llm" ? "LLM agent" : mode === "heuristic-fallback" ? "fallback" : "offline analyst"}</span>}
      </button>
      {open && (
        <ol className="trace-steps">
          {steps.map((s, i) => (
            <li key={i}>
              <div className="trace-head">
                <code>{s.tool}</code>
                <span className="muted">({formatArgs(s.input)})</span>
                <span className="muted right">{s.duration_ms} ms</span>
              </div>
              <pre className="trace-output">{s.output.length > 900 ? s.output.slice(0, 900) + "\n…" : s.output}</pre>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function formatArgs(input: Record<string, unknown>): string {
  return Object.entries(input)
    .filter(([, v]) => v !== "" && v !== null && v !== undefined)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : JSON.stringify(v)}`)
    .join(", ");
}
