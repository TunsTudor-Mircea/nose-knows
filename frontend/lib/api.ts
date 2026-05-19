import type {
  Session,
  Message,
  SessionChatResponse,
  Filters,
  PerfumeListResponse,
  FeedbackListResponse,
  IngestUploadResponse,
  IngestJob,
  IngestDataset,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let message = text || `Request failed with status ${res.status}`;
    try {
      const json = JSON.parse(text);
      if (json.detail) message = json.detail;
    } catch { /* not JSON — use raw text */ }
    throw new Error(message);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── Sessions ──────────────────────────────────────────────────────────────

export const createSession = (title?: string): Promise<Session> =>
  req("/sessions", { method: "POST", body: JSON.stringify({ title: title ?? null }) });

export const listSessions = (): Promise<Session[]> => req("/sessions");

export const deleteSession = (id: string): Promise<void> =>
  req(`/sessions/${id}`, { method: "DELETE" });

export const getMessages = (sessionId: string): Promise<Message[]> =>
  req(`/sessions/${sessionId}/messages`);

// ── Chat ──────────────────────────────────────────────────────────────────

export const sendChat = (
  sessionId: string,
  query: string,
  filters: Filters,
  signal?: AbortSignal,
): Promise<SessionChatResponse> =>
  req(`/sessions/${sessionId}/chat`, {
    method: "POST",
    body: JSON.stringify({ query, filters }),
    signal,
  });

// ── Feedback ──────────────────────────────────────────────────────────────

export const postFeedback = (
  messageId: string,
  score: 1 | -1,
  comment?: string,
): Promise<void> =>
  req(`/messages/${messageId}/feedback`, {
    method: "POST",
    body: JSON.stringify({ score, comment: comment ?? null }),
  });

export const listFeedback = (params: {
  limit?: number;
  offset?: number;
  score?: string;
}): Promise<FeedbackListResponse> => {
  const qs = new URLSearchParams();
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.offset) qs.set("offset", String(params.offset));
  if (params.score) qs.set("score", params.score);
  return req(`/feedback/list?${qs}`);
};

export const deleteFeedback = (id: string): Promise<void> =>
  req(`/feedback/${id}`, { method: "DELETE" });

// ── Perfumes (Explorer) ───────────────────────────────────────────────────

export const listPerfumes = (params: {
  search?: string;
  gender?: string;
  limit?: number;
  offset?: number;
}): Promise<PerfumeListResponse> => {
  const qs = new URLSearchParams();
  if (params.search) qs.set("search", params.search);
  if (params.gender) qs.set("gender", params.gender);
  if (params.limit) qs.set("limit", String(params.limit));
  if (params.offset) qs.set("offset", String(params.offset));
  return req(`/perfumes?${qs}`);
};

// ── Brands ───────────────────────────────────────────────────────────────

export const searchBrands = (q: string): Promise<string[]> =>
  req(`/brands?q=${encodeURIComponent(q)}`);

// ── Ingestion ────────────────────────────────────────────────────────────

export const uploadDataset = async (file: File): Promise<IngestUploadResponse> => {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/ingest/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
};

export const runIngestion = (filename: string): Promise<{ job_id: string; status: string }> =>
  req("/ingest/run", { method: "POST", body: JSON.stringify({ filename }) });

export const getIngestStatus = (jobId: string): Promise<IngestJob> =>
  req(`/ingest/status/${jobId}`);

export const listDatasets = (): Promise<IngestDataset[]> => req("/ingest/datasets");

// ── Eval ─────────────────────────────────────────────────────────────────

export const getEvalResults = (): Promise<EvalResult[]> => req("/eval/results");

export interface EvalResult {
  id: string;
  query: string;
  intent: string;
  response: string;
  retrieved: string;
  hyde_doc: string | null;
  trace: { tool: string; tool_input: string; observation: string }[];
  agent_latency_ms: number;
  scores?: { relevance: number; groundedness: number; helpfulness: number };
  judge_reasoning?: string;
  tag: string;
  run_id: string;
  error?: string;
}

// ── Health ────────────────────────────────────────────────────────────────

export const checkHealth = async (): Promise<boolean> => {
  try {
    const res = await fetch(`${API}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
};
