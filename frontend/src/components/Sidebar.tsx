import { useRef, useState, type DragEvent } from "react";
import type { ColumnProfile, DatasetDetail, DatasetSummary } from "../api";

interface Props {
  open: boolean;
  datasets: DatasetSummary[];
  activeId: string;
  detail: DatasetDetail | null;
  uploading: boolean;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  onDelete: (id: string) => void;
  onAsk: (q: string) => void;
}

const KIND_ICON: Record<ColumnProfile["kind"], string> = {
  numeric: "#",
  categorical: "Aa",
  boolean: "✓",
  datetime: "dt",
  identifier: "ID",
  text: "¶",
};

export function Sidebar({ open, datasets, activeId, detail, uploading, onSelect, onUpload, onDelete, onAsk }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDrag(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <aside className={`sidebar ${open ? "open" : ""}`}>
      <section className="panel">
        <h2>Datasets</h2>
        <ul className="dataset-list">
          {datasets.map((d) => (
            <li key={d.id} className={d.id === activeId ? "active" : ""}>
              <button className="dataset-btn" onClick={() => onSelect(d.id)} title={d.name}>
                <span className="dataset-name">
                  <span className={`src-tag ${d.source}`}>{d.source === "demo" ? "DEMO" : "CSV"}</span>
                  {d.name}
                </span>
                <span className="dataset-meta">
                  {d.rows.toLocaleString()} rows · {d.columns} cols
                </span>
              </button>
              {d.source === "upload" && (
                <button className="icon-btn danger" onClick={() => onDelete(d.id)} aria-label={`Delete ${d.name}`} title="Delete">
                  ✕
                </button>
              )}
            </li>
          ))}
        </ul>
        <div
          className={`dropzone ${drag ? "drag" : ""} ${uploading ? "busy" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onUpload(f);
              e.target.value = "";
            }}
          />
          {uploading ? "Analysing…" : "⬆ Drop a CSV here or click to upload"}
        </div>
      </section>

      {detail && (
        <>
          <section className="panel">
            <h2>Profile</h2>
            <div className="metrics">
              <Metric label="Rows" value={detail.rows.toLocaleString()} />
              <Metric label="Columns" value={String(detail.columns)} />
              <Metric label="Complete" value={`${detail.completeness_pct}%`} />
              <Metric label="Quality" value={`${detail.quality_score}`} accent={detail.quality_score >= 90 ? "good" : detail.quality_score >= 70 ? "warn" : "bad"} />
              <Metric label="Missing" value={detail.missing_cells.toLocaleString()} />
              <Metric label="Duplicates" value={detail.duplicate_rows.toLocaleString()} />
            </div>
          </section>

          <section className="panel columns-panel">
            <h2>
              Columns <span className="muted">({detail.column_profiles.length})</span>
            </h2>
            <ul className="column-list">
              {detail.column_profiles.map((c) => (
                <li key={c.name}>
                  <button
                    className="column-btn"
                    title={`${c.dtype} · ${c.unique} unique · sample: ${c.sample.join(", ")}`}
                    onClick={() => onAsk(c.kind === "numeric" ? `Show the distribution of ${c.name}` : c.kind === "datetime" ? `Monthly trend over ${c.name}` : `Most common values of ${c.name}`)}
                  >
                    <span className={`kind kind-${c.kind}`}>{KIND_ICON[c.kind]}</span>
                    <span className="col-name">{c.name}</span>
                    {c.missing_pct > 0 && <span className="col-missing">{c.missing_pct}% ∅</span>}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </aside>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: "good" | "warn" | "bad" }) {
  return (
    <div className={`metric ${accent ?? ""}`}>
      <span className="metric-value">{value}</span>
      <span className="metric-label">{label}</span>
    </div>
  );
}
