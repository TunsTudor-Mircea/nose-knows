import type { ChatRequest, ChatResponse, FeedbackRequest } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Chat request failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function sendFeedback(req: FeedbackRequest): Promise<void> {
  await fetch(`${API_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
