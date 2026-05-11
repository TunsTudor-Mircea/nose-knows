"use client";
import { useRef, useState } from "react";
import { Upload, Check, ArrowRight, Info, X } from "lucide-react";

const DEST_FIELDS = [
  "Perfume", "Brand", "Year", "Gender",
  "Top", "Middle", "Base",
  "mainaccord1", "mainaccord2", "mainaccord3",
  "Rating Value", "Rating Count", "url",
];

const MAPPING = [
  { src: "perfume_name", dest: "Perfume", confidence: 0.97 },
  { src: "house", dest: "Brand", confidence: 0.94 },
  { src: "release_year", dest: "Year", confidence: 0.89 },
  { src: "category", dest: "Gender", confidence: 0.85 },
  { src: "top_notes_csv", dest: "Top", confidence: 0.91 },
  { src: "middle_notes_csv", dest: "Middle", confidence: 0.90 },
  { src: "base_notes_csv", dest: "Base", confidence: 0.88 },
  { src: "accord_1", dest: "mainaccord1", confidence: 0.82 },
  { src: "accord_2", dest: "mainaccord2", confidence: 0.81 },
  { src: "accord_3", dest: "mainaccord3", confidence: 0.80 },
  { src: "rating_avg", dest: "Rating Value", confidence: 0.93 },
  { src: "n_reviews", dest: "Rating Count", confidence: 0.90 },
  { src: "page_url", dest: "url", confidence: 0.99 },
];

export default function IngestPage() {
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  function handleFile(f: File) {
    setFileName(f.name);
    setStep(1);
    setTimeout(() => setStep(2), 800);
  }

  function startIngest() {
    setRunning(true);
    setProgress(0);
    setStep(3);
    const t = setInterval(() => {
      setProgress((p) => {
        const next = p + 1.5 + Math.random() * 2;
        if (next >= 100) { clearInterval(t); setRunning(false); return 100; }
        return next;
      });
    }, 90);
  }

  const steps = [
    { num: "01", name: "Upload", info: "CSV or JSONL · ≤ 250 MB" },
    { num: "02", name: "Map fields", info: "Auto-detected, review below" },
    { num: "03", name: "Preview", info: "Sample 10 rows, sanity-check" },
    { num: "04", name: "Run batch", info: "Embed and write to ChromaDB" },
  ];

  return (
    <>
      <div className="topbar">
        <div className="crumbs">
          <h1>Ingestion</h1>
          <span className="desc">Upload, map, validate, embed</span>
        </div>
        <div className="topbar-right">
          <span>chunks.jsonl · 24 218</span>
          <span>last batch · 2 d ago</span>
          <button className="btn ghost sm"><Info size={12} /> Pipeline docs</button>
        </div>
      </div>

      <div className="page" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        <div className="stepper">
          {steps.map((s, i) => (
            <div key={s.num} className={`step${i < step ? " done" : ""}${i === step ? " active" : ""}`}>
              <div className="num">{s.num}</div>
              <div className="name">{s.name}</div>
              <div className="info">{s.info}</div>
              {i < step && (
                <span style={{ position: "absolute", top: 14, right: 16, color: "#4e8f6a" }}>
                  <Check size={14} strokeWidth={2.2} />
                </span>
              )}
            </div>
          ))}
        </div>

        {/* Upload */}
        <div className="card" style={{ padding: 26 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <span className="eyebrow">File source</span>
              <h2 className="display" style={{ fontSize: 20, margin: "6px 0 0", fontWeight: 400 }}>
                {fileName ?? "No file selected"}
              </h2>
            </div>
            {fileName && (
              <button className="btn ghost sm" onClick={() => { setFileName(null); setStep(0); }}>
                <X size={12} /> Remove
              </button>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.jsonl"
            style={{ display: "none" }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div
            className="dropzone"
            style={{ padding: 28, cursor: "pointer" }}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]); }}
          >
            <Upload size={26} strokeWidth={1.2} />
            <div className="display" style={{ fontSize: 16, fontWeight: 400 }}>
              Drop a CSV here, or click to browse
            </div>
            <span style={{ fontFamily: "var(--display)", fontSize: 11, letterSpacing: "0.12em", color: "var(--ink-3)" }}>
              The pipeline cleans European decimals, deduplicates by url, then writes chunks.jsonl.
            </span>
          </div>
        </div>

        {/* Mapper */}
        {step >= 2 && (
          <div className="card" style={{ padding: 26 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
              <div>
                <span className="eyebrow">Field mapping · {MAPPING.length} of {MAPPING.length} matched</span>
                <h2 className="display" style={{ fontSize: 20, margin: "6px 0 0", fontWeight: 400 }}>
                  Map source columns to canonical schema
                </h2>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn ghost sm">Reset</button>
                <button className="btn ghost sm">Save as template</button>
              </div>
            </div>
            <div className="mapper">
              {MAPPING.map((m, i) => (
                <>
                  <div key={`src-${i}`} className="src">
                    <span style={{ fontFamily: "var(--display)", fontSize: 10, letterSpacing: "0.14em", color: "var(--ink-3)", textTransform: "uppercase" }}>src</span>
                    <span style={{ marginLeft: 10 }}>{m.src}</span>
                  </div>
                  <div key={`arr-${i}`} className="arr"><ArrowRight size={14} /></div>
                  <div key={`dest-${i}`} className="dest">
                    <select defaultValue={m.dest}>
                      {DEST_FIELDS.map((d) => <option key={d}>{d}</option>)}
                      <option>— skip column —</option>
                    </select>
                    <span className={`status${m.confidence > 0.9 ? " ok" : m.confidence < 0.82 ? " warn" : ""}`}>
                      {(m.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </>
              ))}
            </div>
          </div>
        )}

        {/* Preview + run */}
        {step >= 2 && (
          <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 24 }}>
            <div className="card" style={{ padding: 0, overflow: "hidden" }}>
              <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--grape-line)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span className="eyebrow">Preview · first 6 rows after transform</span>
                <span className="chip lav">all rows valid</span>
              </div>
              <div className="scroll-y" style={{ maxHeight: 280 }}>
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Perfume</th>
                      <th>Brand</th>
                      <th>Year</th>
                      <th>Top</th>
                      <th style={{ textAlign: "right" }}>Rating</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Bois Cellulaire", "Maison Sillage", 2024, "vetiver, paper, lemon", 4.12],
                      ["Or Brûlé", "Atelier Caché", 2023, "saffron, immortelle", 4.31],
                      ["Plage 06h", "Etat Premier", 2025, "salt, fig leaf, neroli", 3.94],
                      ["Velour Noir", "Ombre & Co", 2022, "iris, suede, ambrette", 4.08],
                      ["Tendre Bois", "Maison Sillage", 2024, "white cedar, milk", 4.27],
                      ["Soir d'Encre", "Atelier Caché", 2024, "ink, balsam fir, smoke", 4.05],
                    ].map((r, i) => (
                      <tr key={i}>
                        <td style={{ fontWeight: 500 }}>{r[0]}</td>
                        <td style={{ color: "var(--ink-2)" }}>{r[1]}</td>
                        <td style={{ color: "var(--ink-3)" }}>{r[2]}</td>
                        <td style={{ color: "var(--ink-2)", maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r[3]}</td>
                        <td style={{ textAlign: "right", fontFamily: "var(--display)" }}>{r[4]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card" style={{ padding: 26, display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <span className="eyebrow">Batch run</span>
                <h2 className="display" style={{ fontSize: 20, margin: "6px 0 0", fontWeight: 400 }}>
                  {fileName ?? "No file"} ready
                </h2>
              </div>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 10, fontSize: 13, color: "var(--ink-2)" }}>
                {[
                  ["Decimal commas normalised", true],
                  ["Duplicate URLs dropped", true],
                  ["Brand cased and trimmed", true],
                  ["Embedding model · all-MiniLM-L6-v2", true],
                  ["Estimated time · ~4 min", null],
                ].map((row, i) => (
                  <li key={i} style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>{row[0]}</span>
                    <span style={{ color: row[1] ? "#4e8f6a" : "var(--ink-3)" }}>
                      {row[1] ? <Check size={13} strokeWidth={2.2} /> : "—"}
                    </span>
                  </li>
                ))}
              </ul>
              {(running || progress > 0) && (
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontFamily: "var(--display)", fontSize: 11, letterSpacing: "0.1em", color: "var(--ink-3)" }}>
                    <span>{running ? "Embedding…" : "Done"}</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <div className="bar"><span style={{ width: `${progress}%` }} /></div>
                </div>
              )}
              <div style={{ marginTop: "auto", display: "flex", gap: 8 }}>
                <button className="btn ghost" style={{ flex: 1 }} disabled={running}>Dry run</button>
                <button className="btn primary" style={{ flex: 1 }} onClick={startIngest} disabled={running || !fileName}>
                  {running ? "Running…" : "Run ingestion"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
