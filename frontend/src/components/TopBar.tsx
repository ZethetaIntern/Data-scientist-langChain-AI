import type { Health } from "../api";

interface Props {
  health: Health | null;
  reportUrl: string | null;
  onNewConversation: () => void;
  onToggleSidebar: () => void;
}

export function TopBar({ health, reportUrl, onNewConversation, onToggleSidebar }: Props) {
  const llm = health?.llm;
  const isLlm = llm?.mode === "llm";
  return (
    <header className="topbar">
      <button className="icon-btn menu" onClick={onToggleSidebar} aria-label="Toggle dataset panel">
        ☰
      </button>
      <div className="brand">
        <span className="logo" aria-hidden>
          <svg viewBox="0 0 64 64" width="26" height="26">
            <rect width="64" height="64" rx="14" fill="#0f1420" />
            <rect x="12" y="34" width="8" height="18" rx="2" fill="#4f8cff" />
            <rect x="24" y="24" width="8" height="28" rx="2" fill="#8f7dff" />
            <rect x="36" y="14" width="8" height="38" rx="2" fill="#c77dff" />
            <circle cx="50" cy="14" r="5" fill="#3ddc97" />
          </svg>
        </span>
        <div>
          <h1>Data Scientist LangChain AI</h1>
          <p>Ask questions · get charts, statistics and models</p>
        </div>
      </div>
      <div className="topbar-actions">
        {llm && (
          <span className={`badge ${isLlm ? "badge-llm" : "badge-heuristic"}`} title={llm.detail}>
            <span className="dot" />
            {isLlm ? `LLM · ${llm.provider} / ${llm.model}` : "Offline analyst · no API key"}
          </span>
        )}
        <button className="ghost-btn" onClick={onNewConversation}>
          New chat
        </button>
        {reportUrl && (
          <a className="primary-btn" href={reportUrl} download>
            ⬇ Report
          </a>
        )}
      </div>
    </header>
  );
}
