"use client";
import { useCallback, useEffect, useState } from "react";
import { ThumbsUp, ThumbsDown, Trash2 } from "lucide-react";
import { listFeedback, deleteFeedback } from "@/lib/api";
import type { FeedbackEvent } from "@/lib/types";

function fmt(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export default function FeedbackPage() {
  const [items, setItems] = useState<FeedbackEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [filt, setFilt] = useState<"all" | "up" | "down">("all");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async (filter: "all" | "up" | "down") => {
    setLoading(true);
    try {
      const scoreParam = filter === "up" ? "1" : filter === "down" ? "-1" : undefined;
      const res = await listFeedback({ limit: 50, score: scoreParam });
      setTotal(res.total);
      setItems(res.items);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load("all"); }, [load]);

  function handleFilt(f: "all" | "up" | "down") {
    setFilt(f);
    load(f);
  }

  const handleDelete = useCallback(async (id: string) => {
    try {
      await deleteFeedback(id);
      setItems((prev) => prev.filter((e) => e.id !== id));
      setTotal((t) => t - 1);
    } catch (e) {
      console.error(e);
    }
  }, []);

  const upCount = items.filter((e) => e.score === 1).length;
  const downCount = items.filter((e) => e.score === -1).length;
  const approvalRate = items.length > 0 ? (upCount / items.length).toFixed(2) : "—";

  const intentMap: Record<string, { up: number; total: number }> = {};
  items.forEach((e) => {
    const k = e.intent ?? "unknown";
    if (!intentMap[k]) intentMap[k] = { up: 0, total: 0 };
    intentMap[k].total++;
    if (e.score === 1) intentMap[k].up++;
  });

  return (
    <>
      <div className="page" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        <div className="stat-strip">
          <div className="stat">
            <div className="lbl">Total ratings</div>
            <div className="val">{total.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="lbl">Approval rate</div>
            <div className="val">{approvalRate}</div>
          </div>
          <div className="stat">
            <div className="lbl">Positive</div>
            <div className="val">{upCount}</div>
            <div className="delta up">thumbs up</div>
          </div>
          <div className="stat">
            <div className="lbl">Negative</div>
            <div className="val">{downCount}</div>
            <div className="delta down">thumbs down</div>
          </div>
        </div>

        <div className="fb-grid">
          <div className="card" style={{ padding: 26 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 18 }}>
              <div>
                <span className="eyebrow">Submitted feedback</span>
                <h3 className="display" style={{ fontSize: 18, margin: "6px 0 0", fontWeight: 400 }}>
                  {loading ? "Loading…" : `Showing ${items.length} of ${total}`}
                </h3>
              </div>
              <div className="seg" style={{ width: 280 }}>
                {([["all", "All"], ["up", `👍 ${upCount}`], ["down", `👎 ${downCount}`]] as [typeof filt, string][]).map(([v, l]) => (
                  <button key={v} className={filt === v ? "on" : ""} onClick={() => handleFilt(v)}>{l}</button>
                ))}
              </div>
            </div>

            {items.length === 0 && !loading && (
              <div style={{ padding: "32px 0", color: "var(--ink-3)", fontFamily: "var(--display)", fontSize: 12, letterSpacing: "0.06em", textAlign: "center" }}>
                No feedback yet. Rate some recommendations in the Consult tab.
              </div>
            )}

            <div className="timeline">
              {items.map((e) => (
                <div key={e.id} className="event">
                  <span className="ts">{fmt(e.created_at)}{e.intent ? ` · ${e.intent}` : ""}</span>
                  <div className="body">
                    {e.query && <div className="q">{e.query}</div>}
                    {e.message_content && <div className="resp">{e.message_content}</div>}
                    {e.comment && (
                      <div style={{ marginTop: 6, fontFamily: "var(--display)", fontSize: 11, color: "var(--ink-3)", letterSpacing: "0.04em" }}>
                        "{e.comment}"
                      </div>
                    )}
                  </div>
                  <span className={`verdict ${e.score === 1 ? "up" : "down"}`}>
                    {e.score === 1 ? <ThumbsUp size={16} strokeWidth={1.6} /> : <ThumbsDown size={16} strokeWidth={1.6} />}
                  </span>
                  <button
                    className="btn ghost sm"
                    style={{ padding: "2px 6px", marginLeft: 8, color: "var(--ink-3)" }}
                    onClick={() => handleDelete(e.id)}
                    aria-label="Delete feedback"
                    title="Delete"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="card" style={{ padding: 22 }}>
              <span className="eyebrow">By intent</span>
              <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 14 }}>
                {Object.entries(intentMap).map(([name, { up, total: t }]) => {
                  const rate = t > 0 ? up / t : 0;
                  return (
                    <div key={name}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                        <span style={{ fontFamily: "var(--display)", letterSpacing: "0.04em" }}>{name}</span>
                        <span style={{ color: "var(--ink-3)" }}>{(rate * 100).toFixed(0)}% · {t}</span>
                      </div>
                      <div className="bar"><span style={{ width: `${rate * 100}%` }} /></div>
                    </div>
                  );
                })}
                {Object.keys(intentMap).length === 0 && (
                  <div style={{ color: "var(--ink-3)", fontFamily: "var(--display)", fontSize: 12 }}>No data yet</div>
                )}
              </div>
            </div>

            <div className="card" style={{ padding: 22 }}>
              <span className="eyebrow">Ready for SFT</span>
              <h3 className="display" style={{ fontSize: 16, margin: "6px 0 6px", fontWeight: 400 }}>
                {upCount} positive signal{upCount !== 1 ? "s" : ""}
              </h3>
              <p style={{ fontSize: 12.5, color: "var(--ink-2)", lineHeight: 1.5, margin: "0 0 14px" }}>
                Thumbs-up responses ready to seed the next LoRA round.
              </p>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn ghost sm" style={{ flex: 1 }}>Preview pairs</button>
                <button className="btn primary sm" style={{ flex: 1 }}>Queue training</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
