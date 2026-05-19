"use client";
import { useEffect, useState } from "react";
import { ArrowRight, Check } from "lucide-react";
import { getEvalResults, type EvalResult } from "@/lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseRetrieved(raw: string): { name: string; brand: string; sim: number }[] {
  if (!raw) return [];
  const blocks = raw.split(/\n(?=\d+\.)/).filter(Boolean);
  return blocks.map((block) => {
    const header = block.split("\n")[0];
    const m = header.match(/^\d+\.\s+(.+?)\s+—\s+(.+)$/);
    const simM = block.match(/Distance:\s*([\d.]+)/);
    return {
      name: m?.[1]?.trim() ?? header.trim(),
      brand: m?.[2]?.trim() ?? "",
      sim: simM ? 1 - parseFloat(simM[1]) : 0,
    };
  });
}

function isUsed(name: string, response: string): boolean {
  return response.toLowerCase().includes(name.toLowerCase());
}

function normalise(score: number): number {
  return Math.round((score / 5) * 100) / 100;
}

function ScoreCard({
  label,
  value,
  showPass,
}: {
  label: string;
  value: number;
  showPass?: boolean;
}) {
  return (
    <div className={`score${showPass ? " ok" : ""}`}>
      <div className="lbl">
        {label}
        {showPass && (
          <span className="score-flag pass">
            <Check size={9} strokeWidth={2.4} /> pass
          </span>
        )}
      </div>
      <div className="val">
        {value.toFixed(2)}
        <span className="s">/ 1.00</span>
      </div>
      <div className="bar">
        <span style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function EvalPage() {
  const [allResults, setAllResults] = useState<EvalResult[]>([]);
  const [activeTag, setActiveTag] = useState<string>("");
  const [idx, setIdx] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getEvalResults()
      .then((rows) => {
        const valid = rows.filter((r) => r.scores && r.query);
        setAllResults(valid);
        if (valid.length > 0) setActiveTag(valid[valid.length - 1].tag);
      })
      .finally(() => setLoading(false));
  }, []);

  const tags = [...new Set(allResults.map((r) => r.tag))];
  const results = activeTag ? allResults.filter((r) => r.tag === activeTag) : allResults;
  const row = results[idx] ?? null;
  const retrieved = row ? parseRetrieved(row.retrieved) : [];
  const usedCount = row
    ? retrieved.filter((r) => isUsed(r.name, row.response)).length
    : 0;
  const totalMs = row
    ? row.trace.reduce((a, s) => {
        const m = s.observation?.match(/(\d+)\s*ms/);
        return a + (m ? parseInt(m[1]) : 0);
      }, row.agent_latency_ms ?? 0)
    : 0;

  const rel = row?.scores ? normalise(row.scores.relevance) : 0;
  const grd = row?.scores ? normalise(row.scores.groundedness) : 0;
  const hlp = row?.scores ? normalise(row.scores.helpfulness) : 0;

  return (
    <>
      <div className="topbar">
        <div className="topbar-right">
          <span>llm-judge · gpt-4o-mini</span>
          <span>guardrail · detoxify</span>
          {tags.map((t) => (
            <button
              key={t}
              className={`chip${t === activeTag ? " solid" : ""}`}
              style={{ cursor: "pointer" }}
              onClick={() => { setActiveTag(t); setIdx(0); }}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="page" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        {/* fixture selector */}
        <div className="card" style={{ padding: 22 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div>
              <span className="eyebrow">Test query</span>
              <h2 className="display" style={{ fontSize: 20, margin: "6px 0 0", fontWeight: 400 }}>
                {loading ? "Loading results…" : results.length === 0 ? "No eval results yet — run the CLI eval first." : "Select a fixture to inspect"}
              </h2>
            </div>
            {results.length > 0 && (
              <button className="btn ghost sm" onClick={() => setIdx((idx + 1) % results.length)}>
                Next fixture <ArrowRight size={12} />
              </button>
            )}
          </div>

          {row && (
            <textarea
              className="textarea"
              rows={2}
              value={row.query}
              readOnly
              style={{ fontFamily: "var(--display)", fontSize: 15, lineHeight: 1.5, fontWeight: 300 }}
            />
          )}

          <div style={{ display: "flex", gap: 6, marginTop: 12, flexWrap: "wrap" }}>
            {results.map((r, i) => (
              <button
                key={r.id + i}
                className={`chip${i === idx ? " solid" : ""}`}
                style={{ cursor: "pointer" }}
                onClick={() => setIdx(i)}
                title={r.query}
              >
                {r.id}
              </button>
            ))}
          </div>
        </div>

        {row && (
          <div className="eval-grid">
            {/* reasoning chain */}
            <div className="card" style={{ padding: 26 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
                <div>
                  <span className="eyebrow">LangGraph reasoning chain</span>
                  <h3 className="display" style={{ fontSize: 16, margin: "6px 0 0", fontWeight: 400 }}>
                    {row.trace.length} nodes · {(row.agent_latency_ms / 1000).toFixed(2)} s end to end
                  </h3>
                </div>
                <span className="chip lav">intent · {row.intent}</span>
              </div>
              <div className="chain">
                {row.trace.map((s, i) => {
                  const failed = s.tool === "validate_response" && /^FAIL/i.test(s.observation);
                  const passed = s.tool === "validate_response" && /^PASS/i.test(s.observation);
                  return (
                    <div key={i} className="chain-step done">
                      <span className="marker">{String(i + 1).padStart(2, "0")}</span>
                      <div>
                        <div className="head">
                          <span className="tool">{s.tool}</span>
                          {passed && <span className="score-flag pass">pass</span>}
                          {failed && (
                            <span className="score-flag" style={{ background: "#fde2e2", color: "#b05050", padding: "1px 6px", borderRadius: 4, fontSize: 10 }}>
                              fail
                            </span>
                          )}
                        </div>
                        <div className="out" style={{ whiteSpace: "pre-wrap" }}>{s.observation}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* quality scoring */}
              <div className="card" style={{ padding: 22 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
                  <div>
                    <span className="eyebrow">Quality scoring</span>
                    <h3 className="display" style={{ fontSize: 16, margin: "6px 0 0", fontWeight: 400 }}>
                      LLM-judge assessment
                    </h3>
                  </div>
                  <span className="chip">{row.tag}</span>
                </div>
                <div className="score-grid">
                  <ScoreCard label="Relevance" value={rel} />
                  <ScoreCard label="Grounding" value={grd} />
                  <ScoreCard label="Helpfulness" value={hlp} />
                </div>
              </div>

              {/* judge verdict */}
              <div className="card" style={{ padding: 22 }}>
                <span className="eyebrow">Judge verdict</span>
                <p style={{ fontSize: 13.5, lineHeight: 1.6, color: "var(--ink-2)", margin: "8px 0 0" }}>
                  {row.judge_reasoning ?? "No reasoning available."}
                </p>
                <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 6 }}>
                  {[
                    { label: `Intent classified as ${row.intent}`, ok: true },
                    { label: `Cited ${usedCount} of ${retrieved.length} retrieved candidates`, ok: usedCount > 0 },
                    { label: `Relevance ${row.scores!.relevance}/5`, ok: row.scores!.relevance >= 4 },
                    { label: `Groundedness ${row.scores!.groundedness}/5`, ok: row.scores!.groundedness >= 4 },
                  ].map(({ label, ok }) => (
                    <div key={label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: "var(--ink-2)" }}>
                      <span style={{ color: ok ? "#4e8f6a" : "#b05050" }}>
                        <Check size={13} strokeWidth={2.2} />
                      </span>
                      {label}
                    </div>
                  ))}
                </div>
              </div>

              {/* retrieval used */}
              <div className="card" style={{ padding: 22 }}>
                <span className="eyebrow">Retrieval used</span>
                <h3 className="display" style={{ fontSize: 16, margin: "6px 0 14px", fontWeight: 400 }}>
                  {usedCount} of {retrieved.length} cited in answer
                </h3>
                {retrieved.map((r, i) => {
                  const used = isUsed(r.name, row.response);
                  return (
                    <div
                      key={i}
                      style={{ display: "grid", gridTemplateColumns: "20px 1fr auto auto", gap: 10, padding: "8px 0", borderTop: i ? "1px solid var(--grape-line)" : "none", alignItems: "center" }}
                    >
                      <span style={{ fontFamily: "var(--display)", fontSize: 10, color: "var(--ink-3)" }}>{String(i + 1).padStart(2, "0")}</span>
                      <div>
                        <div style={{ fontSize: 12.5, fontWeight: 500 }}>{r.name}</div>
                        <div style={{ fontSize: 10.5, fontFamily: "var(--display)", letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--ink-3)" }}>{r.brand}</div>
                      </div>
                      <span style={{ fontFamily: "var(--display)", fontSize: 12, color: "var(--ink)" }}>{r.sim.toFixed(2)}</span>
                      <span className={`chip${used ? " solid" : ""}`} style={{ padding: "2px 8px", fontSize: 9.5 }}>
                        {used ? "used" : "skipped"}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
