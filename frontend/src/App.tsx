import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type ChatResponse, type DatasetDetail, type DatasetSummary, type Health } from "./api";
import { TopBar } from "./components/TopBar";
import { Sidebar } from "./components/Sidebar";
import { Chat, type Message } from "./components/Chat";

const SESSION_KEY = "ds-agent:sessions";

function loadSessions(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY) || "{}");
  } catch {
    return {};
  }
}

function newSessionId(): string {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [activeId, setActiveId] = useState<string>("demo");
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const sessions = useRef<Record<string, string>>(loadSessions());

  const sessionFor = useCallback((datasetId: string) => {
    if (!sessions.current[datasetId]) {
      sessions.current[datasetId] = newSessionId();
      localStorage.setItem(SESSION_KEY, JSON.stringify(sessions.current));
    }
    return sessions.current[datasetId];
  }, []);

  const refreshDatasets = useCallback(async () => {
    const list = await api.datasets();
    setDatasets(list);
    return list;
  }, []);

  // initial load
  useEffect(() => {
    (async () => {
      try {
        const [h] = await Promise.all([api.health(), refreshDatasets()]);
        setHealth(h);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not reach the API");
      }
    })();
  }, [refreshDatasets]);

  // load dataset detail + history when the active dataset changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setDetail(null);
      setMessages([]);
      try {
        const d = await api.dataset(activeId);
        if (cancelled) return;
        setDetail(d);
        const sid = sessionFor(activeId);
        const res = await fetch(`/api/datasets/${activeId}/history?session_id=${sid}`);
        if (res.ok && !cancelled) {
          const hist = (await res.json()) as { role: "user" | "assistant"; content: string; charts: ChatResponse["charts"]; steps: ChatResponse["steps"]; suggestions: string[] }[];
          setMessages(hist.map((h, i) => ({ id: `h${i}`, role: h.role, content: h.content, charts: h.charts, steps: h.steps, suggestions: h.suggestions })));
        }
      } catch (e) {
        if (!cancelled) {
          if (e instanceof ApiError && e.status === 404 && activeId !== "demo") {
            setActiveId("demo");
          } else {
            setError(e instanceof Error ? e.message : "Failed to load dataset");
          }
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId, sessionFor]);

  const ask = useCallback(
    async (question: string) => {
      if (!question.trim() || busy || !detail) return;
      setError(null);
      const userMsg: Message = { id: `u${Date.now()}`, role: "user", content: question.trim(), charts: [], steps: [] };
      setMessages((m) => [...m, userMsg]);
      setBusy(true);
      try {
        const reply = await api.chat(detail.id, question.trim(), sessionFor(detail.id));
        setMessages((m) => [
          ...m,
          { id: `a${Date.now()}`, role: "assistant", content: reply.answer, charts: reply.charts, steps: reply.steps, suggestions: reply.suggestions, mode: reply.mode, elapsed_ms: reply.elapsed_ms },
        ]);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "The agent failed";
        setMessages((m) => [...m, { id: `e${Date.now()}`, role: "assistant", content: `**Something went wrong:** ${msg}`, charts: [], steps: [], error: true }]);
      } finally {
        setBusy(false);
      }
    },
    [busy, detail, sessionFor],
  );

  const upload = useCallback(
    async (file: File) => {
      setError(null);
      setUploading(true);
      try {
        const d = await api.upload(file);
        await refreshDatasets();
        setActiveId(d.id);
        setSidebarOpen(false);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [refreshDatasets],
  );

  const remove = useCallback(
    async (id: string) => {
      try {
        await api.remove(id);
        delete sessions.current[id];
        localStorage.setItem(SESSION_KEY, JSON.stringify(sessions.current));
        await refreshDatasets();
        if (id === activeId) setActiveId("demo");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Delete failed");
      }
    },
    [activeId, refreshDatasets],
  );

  const newConversation = useCallback(() => {
    sessions.current[activeId] = newSessionId();
    localStorage.setItem(SESSION_KEY, JSON.stringify(sessions.current));
    setMessages([]);
  }, [activeId]);

  return (
    <div className="app">
      <TopBar
        health={health}
        reportUrl={detail && messages.length ? api.reportUrl(detail.id, sessionFor(detail.id)) : null}
        onNewConversation={newConversation}
        onToggleSidebar={() => setSidebarOpen((o) => !o)}
      />
      <div className="layout">
        <Sidebar
          open={sidebarOpen}
          datasets={datasets}
          activeId={activeId}
          detail={detail}
          uploading={uploading}
          onSelect={(id) => {
            setActiveId(id);
            setSidebarOpen(false);
          }}
          onUpload={upload}
          onDelete={remove}
          onAsk={ask}
        />
        <main className="main">
          {error && (
            <div className="banner error" role="alert">
              <span>{error}</span>
              <button onClick={() => setError(null)} aria-label="Dismiss">
                ×
              </button>
            </div>
          )}
          <Chat detail={detail} health={health} messages={messages} busy={busy} onAsk={ask} />
        </main>
      </div>
    </div>
  );
}
