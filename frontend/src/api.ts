// Thin typed client for the FastAPI backend. All URLs are relative so the same bundle
// works behind the Vite dev proxy, the FastAPI static host and any reverse proxy.

export interface ColumnProfile {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical" | "boolean" | "datetime" | "identifier" | "text";
  missing: number;
  missing_pct: number;
  unique: number;
  sample: string[];
}

export interface DatasetSummary {
  id: string;
  name: string;
  source: "demo" | "upload";
  rows: number;
  columns: number;
  created_at: number;
}

export interface DatasetDetail extends DatasetSummary {
  memory_mb: number;
  missing_cells: number;
  completeness_pct: number;
  duplicate_rows: number;
  quality_score: number;
  column_profiles: ColumnProfile[];
  numeric_columns: string[];
  categorical_columns: string[];
  datetime_columns: string[];
  preview: Record<string, unknown>[];
  suggested_questions: string[];
}

export interface Chart {
  id: string;
  title: string;
  kind: string;
  url: string;
}

export interface Step {
  tool: string;
  input: Record<string, unknown>;
  output: string;
  duration_ms: number;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  steps: Step[];
  charts: Chart[];
  suggestions: string[];
  mode: string;
  elapsed_ms: number;
}

export interface Health {
  status: string;
  version: string;
  llm: { provider: string; model: string | null; mode: "llm" | "heuristic"; detail: string };
  code_tool_enabled: boolean;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  datasets: () => request<DatasetSummary[]>("/api/datasets"),
  dataset: (id: string) => request<DatasetDetail>(`/api/datasets/${id}`),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DatasetDetail>("/api/datasets", { method: "POST", body: form });
  },
  remove: (id: string) => request<void>(`/api/datasets/${id}`, { method: "DELETE" }),
  chat: (id: string, message: string, sessionId: string | null) =>
    request<ChatResponse>(`/api/datasets/${id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    }),
  reportUrl: (id: string, sessionId: string) => `/api/datasets/${id}/report?session_id=${encodeURIComponent(sessionId)}`,
};
