"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Send, Settings2 } from "lucide-react";
import { nanoid } from "nanoid";
import ChatMessage from "@/components/ChatMessage";
import Sidebar from "@/components/Sidebar";
import { checkHealth, sendChat, sendFeedback } from "@/lib/api";
import type { ChatMessage as ChatMessageType, Filters } from "@/lib/types";

const DEFAULT_FILTERS: Filters = {
  gender: "",
  accord: "",
  brand: "",
  top_k: 5,
  use_hyde: true,
};

const EXAMPLE_QUERIES = [
  "Something warm and cozy for a winter date night",
  "A fresh scent for a morning jog in the forest",
  "I want to smell like a powerful CEO walking into a boardroom",
  "Something light and floral for a spring picnic",
];

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Backend health check
  useEffect(() => {
    checkHealth().then(setBackendOnline);
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(
    async (query: string) => {
      if (!query.trim() || loading) return;
      setInput("");
      setLoading(true);

      const userMsg: ChatMessageType = {
        id: nanoid(),
        role: "user",
        content: query.trim(),
      };
      setMessages((prev) => [...prev, userMsg]);

      try {
        const resp = await sendChat({
          query: query.trim(),
          filters: {
            gender: filters.gender || undefined,
            accord: filters.accord || undefined,
            brand: filters.brand || undefined,
            top_k: filters.top_k,
            use_hyde: filters.use_hyde,
          },
        });

        const assistantMsg: ChatMessageType = {
          id: nanoid(),
          role: "assistant",
          content: resp.response,
          perfumes: resp.perfumes,
          hyde_doc: resp.hyde_doc,
          intent: resp.intent,
          feedback: null,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const errorMsg: ChatMessageType = {
          id: nanoid(),
          role: "assistant",
          content: `Something went wrong: ${err instanceof Error ? err.message : String(err)}`,
          feedback: null,
        };
        setMessages((prev) => [...prev, errorMsg]);
      } finally {
        setLoading(false);
      }
    },
    [loading, filters]
  );

  const handleFeedback = useCallback(
    async (messageId: string, score: 1 | -1) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, feedback: score } : m))
      );

      // Find the message and its preceding user query
      const msgIndex = messages.findIndex((m) => m.id === messageId);
      const assistantMsg = messages[msgIndex];
      const userMsg = messages
        .slice(0, msgIndex)
        .reverse()
        .find((m) => m.role === "user");

      if (assistantMsg && userMsg) {
        await sendFeedback({
          query: userMsg.content,
          response: assistantMsg.content,
          score: score === 1 ? 5 : 1,
        }).catch(console.error);
      }
    },
    [messages]
  );

  return (
    <div className="flex h-screen bg-stone-50 font-sans">
      {/* Main content */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 bg-white border-b border-stone-200 shadow-sm">
          <div className="flex items-center gap-3">
            <span className="text-2xl" aria-hidden>
              🌸
            </span>
            <div>
              <h1 className="text-lg font-bold text-gray-900 leading-none">NoseKnows</h1>
              <p className="text-xs text-gray-400">AI Fragrance Consultant</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Backend status indicator */}
            <div className="hidden sm:flex items-center gap-1.5">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendOnline === null
                    ? "bg-gray-300 animate-pulse"
                    : backendOnline
                    ? "bg-green-400"
                    : "bg-red-400"
                }`}
              />
              <span className="text-xs text-gray-400">
                {backendOnline === null ? "Connecting…" : backendOnline ? "Online" : "Backend offline"}
              </span>
            </div>

            <button
              onClick={() => setSidebarOpen(true)}
              className="flex items-center gap-1.5 text-sm text-gray-600 hover:text-indigo-600 transition-colors md:hidden"
              aria-label="Open settings"
            >
              <Settings2 className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 flex flex-col gap-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center flex-1 gap-6 py-12">
              <div className="text-center">
                <p className="text-4xl mb-3">🌿</p>
                <h2 className="text-xl font-semibold text-gray-700">
                  Describe your perfect scent
                </h2>
                <p className="text-sm text-gray-400 mt-1 max-w-sm">
                  Tell me about a mood, occasion, or feeling and I&apos;ll find the right fragrance for you.
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
                {EXAMPLE_QUERIES.map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    className="text-left text-sm text-gray-600 bg-white border border-stone-200 rounded-xl px-4 py-3 hover:border-indigo-400 hover:text-indigo-700 transition-colors shadow-sm"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} onFeedback={handleFeedback} />
          ))}

          {loading && (
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>NoseKnows is thinking…</span>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div className="px-4 py-4 bg-white border-t border-stone-200">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend(input);
            }}
            className="flex items-end gap-3 max-w-3xl mx-auto"
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(input);
                }
              }}
              placeholder="Describe the scent you're looking for…"
              rows={1}
              className="flex-1 resize-none rounded-2xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 bg-stone-50 max-h-32"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="p-3 rounded-2xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
              aria-label="Send"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </form>
          <p className="text-center text-xs text-gray-300 mt-2">
            Press Enter to send · Shift+Enter for new line
          </p>
        </div>
      </div>

      {/* Sidebar */}
      <Sidebar
        filters={filters}
        onChange={setFilters}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
    </div>
  );
}
