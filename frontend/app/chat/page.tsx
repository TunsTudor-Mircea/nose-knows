"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Plus, X, ThumbsUp, ThumbsDown, Send, StopCircle } from "lucide-react";
import Markdown from "@/components/Markdown";
import PerfumeCard from "@/components/PerfumeCard";
import {
  createSession, listSessions, deleteSession,
  getMessages, sendChat, postFeedback,
} from "@/lib/api";
import type { Session, Message, Filters } from "@/lib/types";

const DEFAULT_FILTERS: Filters = {
  gender: "any", accord: "any", brand: "", top_k: 5, use_hyde: true,
};

const STARTERS = [
  "Something warm for a winter date night",
  "Powdery iris and suede",
  "Office-friendly fresh",
  "What does Tom Ford pair with patchouli?",
];

const FB_STORAGE_KEY = "nosknows_feedback";

function loadFeedbackCache(): Record<string, 1 | -1> {
  try {
    const raw = localStorage.getItem(FB_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveFeedbackCache(fb: Record<string, 1 | -1>) {
  try { localStorage.setItem(FB_STORAGE_KEY, JSON.stringify(fb)); } catch {}
}

function fmt(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  // Track which session ID is currently waiting for a response (null = idle).
  const [thinkingFor, setThinkingFor] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<Record<string, 1 | -1>>({});
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const streamRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Always reflects the latest activeId inside async callbacks without stale closure.
  const activeIdRef = useRef<string | null>(null);

  // Load sessions on mount
  useEffect(() => {
    listSessions().then((s) => {
      setSessions(s);
      if (s.length > 0) setActiveId(s[0].id);
    }).catch(console.error);
    setFeedback(loadFeedbackCache());
  }, []);

  // Keep ref in sync so async callbacks always see the current session.
  useEffect(() => { activeIdRef.current = activeId; }, [activeId]);

  // Load messages when active session changes
  useEffect(() => {
    if (!activeId) { setMessages([]); return; }
    getMessages(activeId).then(setMessages).catch(console.error);
  }, [activeId]);

  // Scroll to bottom on new messages or while thinking in active session
  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [messages, thinkingFor]);

  const isThinking = thinkingFor !== null;
  const thinkingHere = thinkingFor === activeId;

  const handleNewSession = useCallback(async () => {
    try {
      const s = await createSession();
      setSessions((prev) => [s, ...prev]);
      setActiveId(s.id);
      setMessages([]);
    } catch (e) { console.error(e); }
  }, []);

  const handleDeleteSession = useCallback(async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
      }
    } catch (e) { console.error(e); }
  }, [activeId]);

  const handleCancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setThinkingFor(null);
    // Remove the optimistic user message that was added
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.id.startsWith("tmp-")) return prev.slice(0, -1);
      return prev;
    });
  }, []);

  const handleSend = useCallback(async () => {
    const q = draft.trim();
    if (!q || isThinking) return;

    let sid = activeId;
    if (!sid) {
      try {
        const s = await createSession(q.slice(0, 60));
        setSessions((prev) => [s, ...prev]);
        setActiveId(s.id);
        sid = s.id;
      } catch (e) { console.error(e); return; }
    }

    // Optimistic user message
    const tempUser: Message = {
      id: `tmp-${Date.now()}`,
      session_id: sid,
      role: "user",
      content: q,
      intent: null,
      hyde_doc: null,
      perfumes: null,
      latency_ms: null,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUser]);
    setDraft("");
    setThinkingFor(sid);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const f: Filters = {
        ...filters,
        gender: filters.gender === "any" ? "" : filters.gender,
        accord: filters.accord === "any" ? "" : filters.accord,
      };
      const resp = await sendChat(sid, q, f, controller.signal);

      const asst: Message = {
        id: resp.message_id,
        session_id: sid,
        role: "assistant",
        content: resp.response,
        intent: resp.intent,
        hyde_doc: resp.hyde_doc,
        perfumes: resp.perfumes,
        latency_ms: null,
        created_at: new Date().toISOString(),
      };
      // Only append to the displayed messages if the user is still viewing this session.
      if (activeIdRef.current === sid) {
        setMessages((prev) => [...prev, asst]);
      }

      setSessions((prev) => prev.map((s) =>
        s.id === sid ? { ...s, title: s.title ?? q.slice(0, 60) } : s
      ));
    } catch (e) {
      if ((e as Error).name === "AbortError") return; // user cancelled — already cleaned up
      console.error(e);
      const errMsg: Message = {
        id: `err-${Date.now()}`,
        session_id: sid,
        role: "assistant",
        content: `Something went wrong: ${e instanceof Error ? e.message : String(e)}`,
        intent: null,
        hyde_doc: null,
        perfumes: null,
        latency_ms: null,
        created_at: new Date().toISOString(),
      };
      if (activeIdRef.current === sid) {
        setMessages((prev) => [...prev, errMsg]);
      }
    } finally {
      abortRef.current = null;
      setThinkingFor(null);
    }
  }, [draft, isThinking, activeId, filters]);

  const handleFeedback = useCallback(async (messageId: string, score: 1 | -1) => {
    const current = feedback[messageId];
    // Toggle off if same score pressed again
    const newScore = current === score ? undefined : score;

    setFeedback((f) => {
      const next = { ...f };
      if (newScore === undefined) delete next[messageId];
      else next[messageId] = newScore;
      saveFeedbackCache(next);
      return next;
    });

    if (newScore !== undefined) {
      // Always post (backend upserts, so repeated clicks just update score)
      postFeedback(messageId, newScore).catch(console.error);
    }
  }, [feedback]);

  return (
    <>
      <div className="topbar">
        <div className="crumbs">
          <h1>Consult</h1>
          <span className="desc">Live recommendation chat</span>
        </div>
      </div>

      <div className="page">
        <div className="chat-layout">
          {/* Sessions panel */}
          <div className="sessions-panel">
            <div className="sessions-header">
              <button className="btn primary sm" onClick={handleNewSession}>
                <Plus size={12} /> New chat
              </button>
            </div>
            <div className="sessions-list scroll-y">
              {sessions.length === 0 && (
                <div style={{ padding: "16px", color: "var(--ink-3)", fontSize: 12, fontFamily: "var(--display)", letterSpacing: "0.06em" }}>
                  No sessions yet
                </div>
              )}
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={`session-item${activeId === s.id ? " active" : ""}`}
                  onClick={() => setActiveId(s.id)}
                >
                  <div>
                    <div className="s-title">{s.title ?? "New chat"}</div>
                    <div className="s-time">{fmt(s.updated_at)}</div>
                  </div>
                  <button className="s-del" onClick={(e) => handleDeleteSession(s.id, e)} aria-label="Delete session">
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Chat column */}
          <div className="chat-col">
            <div className="chat-stream scroll-y" ref={streamRef}>
              {messages.length === 0 && !thinkingHere && (
                <div className="empty-state">
                  <div style={{ fontSize: 32 }}>🌿</div>
                  <div>Describe a mood, occasion, or notes you love</div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, width: "100%", maxWidth: 360 }}>
                    {STARTERS.map((s) => (
                      <button key={s} className="chip" style={{ justifyContent: "flex-start", padding: "10px 14px", cursor: "pointer" }}
                        onClick={() => setDraft(s)}>{s}</button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((m) => m.role === "user" ? (
                <div key={m.id} className="msg-user">{m.content}</div>
              ) : (
                <div key={m.id} className="msg-agent">
                  <div className="author">
                    <span className="badge">NoseKnows</span>
                    {m.intent && <span>intent · {m.intent}</span>}
                    {m.hyde_doc && <span>HyDE</span>}
                    {m.latency_ms && <span>{(m.latency_ms / 1000).toFixed(2)} s</span>}
                  </div>
                  <Markdown className="body">{m.content}</Markdown>
                  {m.perfumes && m.perfumes.length > 0 && (
                    <div className="pcards">
                      {m.perfumes.map((p, j) => (
                        <PerfumeCard key={`${m.id}-${j}`} card={p} index={j + 1} />
                      ))}
                    </div>
                  )}
                  <div className="fb">
                    <span>Was this useful?</span>
                    <button
                      className={`up${feedback[m.id] === 1 ? " on" : ""}`}
                      onClick={() => handleFeedback(m.id, 1)}
                      aria-label="Thumbs up"
                    >
                      <ThumbsUp size={13} />
                    </button>
                    <button
                      className={`down${feedback[m.id] === -1 ? " on" : ""}`}
                      onClick={() => handleFeedback(m.id, -1)}
                      aria-label="Thumbs down"
                    >
                      <ThumbsDown size={13} />
                    </button>
                  </div>
                </div>
              ))}

              {/* Thinking indicator only shown in the session that sent the request */}
              {thinkingHere && (
                <div className="msg-agent">
                  <div className="author">
                    <span className="badge">NoseKnows</span>
                    <span>reasoning</span>
                  </div>
                  <span className="thinking">
                    <span className="thinking-dot" />
                    <span className="thinking-dot" />
                    <span className="thinking-dot" />
                    classify · hyde · retrieve · recommend · validate
                  </span>
                </div>
              )}
            </div>

            <div className="chat-input-wrap">
              <div className="composer">
                <textarea
                  rows={1}
                  placeholder="Describe a mood, occasion, or notes you love..."
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                />
                {isThinking ? (
                  <button className="send cancel" onClick={handleCancel} title="Cancel">
                    <StopCircle size={13} /> Cancel
                  </button>
                ) : (
                  <button className="send" disabled={!draft.trim()} onClick={handleSend}>
                    Send <Send size={13} />
                  </button>
                )}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8, fontFamily: "var(--display)", fontSize: 10.5, letterSpacing: "0.12em", color: "var(--ink-3)" }}>
                <span>Shift + Enter for newline</span>
                <span>{filters.use_hyde ? "HyDE on" : "HyDE off"} · top-{filters.top_k} · {filters.gender}</span>
              </div>
            </div>
          </div>

          {/* Filters panel */}
          <aside className="filters">
            <div>
              <h3>Gender</h3>
              <div className="seg">
                {["any", "women", "men"].map((g) => (
                  <button key={g} className={filters.gender === g ? "on" : ""}
                    onClick={() => setFilters((f) => ({ ...f, gender: g }))}>{g}</button>
                ))}
              </div>
            </div>
            <div>
              <h3>Top K candidates</h3>
              <div className="slider-wrap">
                <input className="slider" type="range" min={1} max={20}
                  value={filters.top_k}
                  onChange={(e) => setFilters((f) => ({ ...f, top_k: +e.target.value }))} />
                <span className="slider-val">{filters.top_k}</span>
              </div>
            </div>
            <div>
              <div className="toggle">
                <h3 style={{ margin: 0 }}>HyDE retrieval</h3>
                <button className={`switch${filters.use_hyde ? " on" : ""}`}
                  onClick={() => setFilters((f) => ({ ...f, use_hyde: !f.use_hyde }))} />
              </div>
              <p style={{ fontSize: 11, color: "var(--ink-3)", margin: "8px 0 0", lineHeight: 1.5 }}>
                Generates a hypothetical perfume profile before searching ChromaDB.
              </p>
            </div>
            <div>
              <h3>Brand</h3>
              <input className="input" placeholder="Type a brand..."
                value={filters.brand}
                onChange={(e) => setFilters((f) => ({ ...f, brand: e.target.value }))} />
            </div>
            <div style={{ marginTop: "auto", paddingTop: 18, borderTop: "1px solid var(--grape-line)" }}>
              <h3>Try a starter</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
                {STARTERS.map((s) => (
                  <button key={s} className="chip"
                    style={{ justifyContent: "flex-start", textAlign: "left", padding: "8px 12px", cursor: "pointer" }}
                    onClick={() => setDraft(s)}>{s}</button>
                ))}
              </div>
            </div>
          </aside>
        </div>
      </div>
    </>
  );
}
