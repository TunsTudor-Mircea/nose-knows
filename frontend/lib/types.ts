export interface PerfumeCard {
  perfume: string;
  brand: string;
  top_notes: string[];
  heart_notes: string[];
  base_notes: string[];
  accords: string[];
  rating: number | null;
  url: string;
}

export interface Session {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  intent: string | null;
  hyde_doc: string | null;
  perfumes: PerfumeCard[] | null;
  latency_ms: number | null;
  created_at: string;
}

export interface SessionChatResponse {
  message_id: string;
  response: string;
  perfumes: PerfumeCard[];
  hyde_doc: string | null;
  intent: string | null;
}

export interface Filters {
  top_k: number;
}

export interface PerfumeListItem {
  id: string;
  perfume: string;
  brand: string;
  gender: string | null;
  year: number | null;
  top_notes: string[];
  heart_notes: string[];
  base_notes: string[];
  accords: string[];
  rating: number | null;
  rating_count: number | null;
  url: string;
}

export interface PerfumeListResponse {
  total: number;
  perfumes: PerfumeListItem[];
}

export interface FeedbackEvent {
  id: string;
  message_id: string;
  session_id: string;
  score: 1 | -1;
  comment: string | null;
  created_at: string;
  message_content?: string;
  query?: string;
  intent?: string | null;
}

export interface FeedbackListResponse {
  total: number;
  items: FeedbackEvent[];
}

export interface IngestUploadResponse {
  filename: string;
  detected_columns: string[];
  missing_required: string[];
  row_count: number;
  preview: Record<string, string>[];
  valid: boolean;
}

export interface IngestJob {
  id: string;
  filename: string;
  status: "pending" | "running" | "done" | "error";
  total_rows: number | null;
  processed_rows: number | null;
  progress_pct: number;
  message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface IngestDataset {
  id: string;
  filename: string;
  status: string;
  total_rows: number | null;
  created_at: string;
}
