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

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  perfumes?: PerfumeCard[];
  hyde_doc?: string | null;
  intent?: string | null;
  feedback?: 1 | -1 | null;
}

export interface Filters {
  gender: string;
  accord: string;
  brand: string;
  top_k: number;
  use_hyde: boolean;
}

export interface ChatRequest {
  query: string;
  filters: {
    gender?: string;
    accord?: string;
    brand?: string;
    top_k: number;
    use_hyde: boolean;
  };
}

export interface ChatResponse {
  response: string;
  perfumes: PerfumeCard[];
  hyde_doc: string | null;
  intent: string | null;
}

export interface FeedbackRequest {
  query: string;
  response: string;
  score: number;
}
