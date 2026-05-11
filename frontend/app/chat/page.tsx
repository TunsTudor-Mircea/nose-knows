"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Plus, X, ThumbsUp, ThumbsDown, Send } from "lucide-react";
import ReactMarkdown from "react-markdown";
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

function fmt(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function ChatPage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState(false);
  const [feedback, setFeedback] = useState<Record<string, 1 | -1>>({});
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const streamRef = useRef<HTMLDivElement>(null);

  // Load sessions on mount
  useEffect(() => {
    listSessions().then((s) => {
      setSessions(s);
      if (s.length > 0) {
        setActiveId(s[0].id);
      }
    }).catch(console.error);
  }, []);

  // Load messages when active session changes
  useEffect(() => {
    if (!activeId) { setMessages([]); return; }
    getMessages(activeId).then(setMessages).catch(console.error);
  }, [activeId]);

  // Scroll to bottom
  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [messages, thinking]);

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

  const handleSend = useCallback(async () => {
    const q = draft.trim();
    if (!q || thinking) return;

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
    setThinking(true);

    try {
      const f: Filters = {
        ...filters,
        gender: filters.gender === "any" ? "" : filters.gender,
        accord: filters.accord === "any" ? "" : filters.accord,
      };
      const resp = await sendChat(sid, q, f);

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
      setMessages((prev) => [...prev, asst]);

      // Update session title if it was auto-set
      setSessions((prev) => prev.map((s) =>
        s.id === sid ? { ...s, title: s.title ?? q.slice(0, 60) } : s
      ));
    } catch (e) {
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
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setThinking(false);
    }
  }, [draft, thinking, activeId, filters]);

  const handleFeedback = useCallback(async (messageId: string, score: 1 | -1) => {
    const current = feedback[messageId];
    const newScore = current === score ? undefined : score;
    setFeedback((f) => {
      const next = { ...f };
      if (newScore === undefined) delete next[messageId];
      else next[messageId] = newScore;
      return next;
    });
    if (newScore !== undefined) {
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
              {messages.length === 0 && !thinking && (
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
                  <div className="body"><ReactMarkdown>{m.content}</ReactMarkdown></div>
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

              {thinking && (
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
                  placeholder="Describe a mood, occasion, or notes you love..."
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                />
                <button className="send" disabled={!draft.trim() || thinking} onClick={handleSend}>
                  Send <Send size={13} />
                </button>
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
