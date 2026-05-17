"use client";
import { useState } from "react";
import { ArrowRight, Check } from "lucide-react";

const EVAL_QUERIES = [
  "I want something cozy for a winter date night, warm but not too sweet.",
  "Office friendly, fresh and clean, no oud.",
  "Beach wedding scent for the bride.",
  "Something with rose and saffron, for evening wear.",
];

const CHAIN = [
  { tool: "classify_intent", ms: 312, output: "occasion_based", marker: "01" },
  {
    tool: "generate_hyde_document", ms: 1042, marker: "02",
    output: "Top notes: pink pepper, clove, mandarin\nHeart notes: chestnut, guaiac wood, juniper\nBase notes: vanilla, peru balsam, cashmeran\nAccords: warm spicy, smoky, balsamic, vanilla\nMood: an enveloping, candle-lit warmth that feels intimate without becoming cloying.",
  },
  {
    tool: "retrieve_fragrances", ms: 487, marker: "03",
    output: "1. Replica By the Fireplace — Maison Margiela\n   Top: pink pepper, orange blossom, clove\n   Heart: chestnut, guaiac wood, juniper\n   Base: cashmeran, vanilla, peru balsam\n2. Tobacco Vanille — Tom Ford\n   Top: tobacco leaf, spicy notes\n   Heart: tonka, tobacco blossom, vanilla\n   Base: dried fruits, cocoa, woody\n3. Portrait of a Lady — Frederic Malle\n   Top: raspberry, blackcurrant, cinnamon\n   Heart: turkish rose, clove\n   Base: patchouli, sandalwood, incense, musk",
  },
  {
    tool: "generate_recommendation", ms: 998, marker: "04",
    output: "For a winter date night, you want a fragrance that wraps around you without becoming a dessert. Replica By the Fireplace sits closest to what you described — its chestnut and guaiac core feel like a roasting fire, while pink pepper and clove keep it from going gourmand.",
  },
  {
    tool: "validate_response", ms: 214, marker: "05",
    output: "PASS · grounding score 0.92 · domain content present · toxicity 0.01",
  },
];

const RETRIEVED = [
  { name: "Replica By the Fireplace", brand: "Maison Margiela", sim: 0.81, used: true },
  { name: "Tobacco Vanille", brand: "Tom Ford", sim: 0.76, used: true },
  { name: "Portrait of a Lady", brand: "Frederic Malle", sim: 0.71, used: true },
  { name: "L'Heure Bleue", brand: "Guerlain", sim: 0.68, used: false },
  { name: "Black Phantom", brand: "Kilian", sim: 0.66, used: false },
];

export default function EvalPage() {
  const [qIdx, setQIdx] = useState(0);
  const [running, setRunning] = useState(false);

  function rerun() {
    setRunning(true);
    setTimeout(() => setRunning(false), 1500);
  }

  const query = EVAL_QUERIES[qIdx];
  const totalMs = CHAIN.reduce((a, c) => a + c.ms, 0);

  return (
    <>
      <div className="topbar">
        <div className="topbar-right">
          <span>llm-judge · gemma2:2b</span>
          <span>guardrail · detoxify</span>
          <button className="btn ghost sm">Export trace</button>
        </div>
      </div>

      <div className="page" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        <div className="card" style={{ padding: 22 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div>
              <span className="eyebrow">Test query</span>
              <h2 className="display" style={{ fontSize: 20, margin: "6px 0 0", fontWeight: 400 }}>
                Replay a fixture or paste a new prompt
              </h2>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn ghost sm" onClick={() => setQIdx((qIdx + 1) % EVAL_QUERIES.length)}>
                Next fixture
              </button>
              <button className="btn primary sm" onClick={rerun} disabled={running}>
                {running ? "Tracing…" : <>Run trace <ArrowRight size={12} /></>}
              </button>
            </div>
          </div>
          <textarea
            className="textarea"
            rows={2}
            value={query}
            readOnly
            style={{ fontFamily: "var(--display)", fontSize: 15, lineHeight: 1.5, fontWeight: 300 }}
          />
          <div style={{ display: "flex", gap: 6, marginTop: 12, flexWrap: "wrap" }}>
            {EVAL_QUERIES.map((q, i) => (
              <button
                key={i}
                className={`chip${i === qIdx ? " solid" : ""}`}
                style={{ cursor: "pointer" }}
                onClick={() => setQIdx(i)}
              >
                fixture {String(i + 1).padStart(2, "0")}
              </button>
            ))}
          </div>
        </div>

        <div className="eval-grid">
          <div className="card" style={{ padding: 26 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
              <div>
                <span className="eyebrow">LangGraph reasoning chain</span>
                <h3 className="display" style={{ fontSize: 16, margin: "6px 0 0", fontWeight: 400 }}>
                  {CHAIN.length} nodes · {(totalMs / 1000).toFixed(2)} s end to end
                </h3>
              </div>
              <span className="chip lav">structured fallback · off</span>
            </div>
            <div className="chain">
              {CHAIN.map((s, i) => (
                <div key={i} className="chain-step done">
                  <span className="marker">{s.marker}</span>
                  <div>
                    <div className="head">
                      <span className="tool">{s.tool}</span>
                      <span className="ms">{s.ms} ms</span>
                      {s.tool === "validate_response" && <span className="score-flag pass">pass</span>}
                    </div>
                    <div className="out" style={{ whiteSpace: "pre-wrap" }}>{s.output}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="card" style={{ padding: 22 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 14 }}>
                <div>
                  <span className="eyebrow">Quality scoring</span>
                  <h3 className="display" style={{ fontSize: 16, margin: "6px 0 0", fontWeight: 400 }}>
                    LLM-judge assessment
                  </h3>
                </div>
                <span className="chip">judge · self-host</span>
              </div>
              <div className="score-grid">
                {[
                  { lbl: "Toxicity", val: 0.01, pass: true },
                  { lbl: "Hallucination", val: 0.08, pass: true },
                  { lbl: "Relevance", val: 0.92, pass: false },
                  { lbl: "Grounding", val: 0.92, pass: false },
                ].map(({ lbl, val, pass }) => (
                  <div key={lbl} className={`score${pass ? " ok" : ""}`}>
                    <div className="lbl">
                      {lbl}
                      {pass && <span className="score-flag pass"><Check size={9} strokeWidth={2.4} /> pass</span>}
                    </div>
                    <div className="val">{val.toFixed(2)}<span className="s">/ 1.00</span></div>
                    <div className="bar"><span style={{ width: `${val * 100}%` }} /></div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card" style={{ padding: 22 }}>
              <span className="eyebrow">Judge verdict</span>
              <p style={{ fontSize: 13.5, lineHeight: 1.6, color: "var(--ink-2)", margin: "8px 0 0" }}>
                Recommendation is well-grounded in the retrieved context. All three named perfumes
                appear verbatim in the candidate list. The response stays on-domain and addresses
                the warmth-without-sweetness constraint explicitly. No flags raised.
              </p>
              <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 6 }}>
                {[
                  "Domain content present",
                  "Mentioned 3 of 3 retrieved candidates",
                  "Cited specific notes (chestnut, guaiac wood, pink pepper)",
                  "No hallucinated brands",
                ].map((row) => (
                  <div key={row} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: "var(--ink-2)" }}>
                    <span style={{ color: "#4e8f6a" }}><Check size={13} strokeWidth={2.2} /></span>
                    {row}
                  </div>
                ))}
              </div>
            </div>

            <div className="card" style={{ padding: 22 }}>
              <span className="eyebrow">Retrieval used</span>
              <h3 className="display" style={{ fontSize: 16, margin: "6px 0 14px", fontWeight: 400 }}>
                Top 3 of 5 cited in answer
              </h3>
              {RETRIEVED.map((r, i) => (
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
                  <span className={`chip${r.used ? " solid" : ""}`} style={{ padding: "2px 8px", fontSize: 9.5 }}>
                    {r.used ? "used" : "skipped"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
