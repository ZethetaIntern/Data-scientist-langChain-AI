import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import type { Chart, DatasetDetail, Health, Step } from "../api";
import { Avatar, MessageView } from "./Message";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  charts: Chart[];
  steps: Step[];
  suggestions?: string[];
  mode?: string;
  elapsed_ms?: number;
  error?: boolean;
}

interface Props {
  detail: DatasetDetail | null;
  health: Health | null;
  messages: Message[];
  busy: boolean;
  onAsk: (q: string) => void;
}

export function Chat({ detail, health, messages, busy, onAsk }: Props) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, busy]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (!draft.trim() || busy) return;
    onAsk(draft);
    setDraft("");
    inputRef.current?.focus();
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <section className="chat">
      <div className="messages" aria-live="polite">
        {detail && messages.length === 0 && (
          <div className="intro">
            <h2>
              Analysing <span className="accent">{detail.name}</span>
            </h2>
            <p>
              {detail.rows.toLocaleString()} rows × {detail.columns} columns · {detail.numeric_columns.length} numeric, {detail.categorical_columns.length} categorical
              {detail.datetime_columns.length ? `, ${detail.datetime_columns.length} date` : ""} column{detail.columns === 1 ? "" : "s"}.
              {health?.llm.mode === "llm"
                ? " The LangChain agent will pick the right analysis tools for each question."
                : " Running in offline mode: answers are computed deterministically with pandas & scikit-learn."}
            </p>
            <div className="suggestions">
              {detail.suggested_questions.map((q) => (
                <button key={q} className="chip" onClick={() => onAsk(q)} disabled={busy}>
                  {q}
                </button>
              ))}
            </div>
            <PreviewTable rows={detail.preview} />
          </div>
        )}
        {!detail && <div className="intro skeleton">Loading dataset…</div>}

        {messages.map((m) => (
          <MessageView key={m.id} message={m} onAsk={onAsk} busy={busy} />
        ))}

        {busy && (
          <div className="message assistant thinking">
            <Avatar user={false} />
            <div className="bubble">
              <span className="dots">
                <i />
                <i />
                <i />
              </span>
              Running analysis tools…
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {detail && messages.length > 0 && (
        <div className="suggestions compact">
          {detail.suggested_questions.slice(0, 5).map((q) => (
            <button key={q} className="chip small" onClick={() => onAsk(q)} disabled={busy}>
              {q}
            </button>
          ))}
        </div>
      )}

      <form className="composer" onSubmit={submit}>
        <textarea
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
          placeholder={detail ? `Ask anything about ${detail.name}… e.g. "average revenue by region"` : "Loading…"}
          rows={1}
          disabled={!detail || busy}
          aria-label="Ask the data scientist agent"
        />
        <button type="submit" className="primary-btn send" disabled={!detail || busy || !draft.trim()}>
          {busy ? "…" : "Ask ➤"}
        </button>
      </form>
    </section>
  );
}

function PreviewTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return null;
  const cols = Object.keys(rows[0]);
  return (
    <div className="preview">
      <div className="preview-title">First rows</div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {cols.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 5).map((r, i) => (
              <tr key={i}>
                {cols.map((c) => (
                  <td key={c}>{formatCell(r[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  if (typeof v === "string" && /^\d{4}-\d{2}-\d{2}T/.test(v)) return v.slice(0, 10);
  return String(v);
}
