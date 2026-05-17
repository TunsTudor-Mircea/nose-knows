"use client";
import { useEffect, useRef, useState } from "react";
import { Upload, Check, Info, X, AlertCircle, RefreshCw } from "lucide-react";
import {
  uploadDataset,
  runIngestion,
  getIngestStatus,
  listDatasets,
} from "@/lib/api";
import type { IngestUploadResponse, IngestJob, IngestDataset } from "@/lib/types";

export default function IngestPage() {
  const [step, setStep] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadInfo, setUploadInfo] = useState<IngestUploadResponse | null>(null);
  const [currentJob, setCurrentJob] = useState<IngestJob | null>(null);
  const [pastJobs, setPastJobs] = useState<IngestDataset[]>([]);
  const [loadingPast, setLoadingPast] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadPastJobs();
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function loadPastJobs() {
    setLoadingPast(true);
    try {
      const jobs = await listDatasets();
      setPastJobs(jobs);
    } catch {
      // silently ignore
    } finally {
      setLoadingPast(false);
    }
  }

  async function handleFile(file: File) {
    setFileName(file.name);
    setUploadError(null);
    setUploadInfo(null);
    setStep(1);
    setUploading(true);
    try {
      const info = await uploadDataset(file);
      setUploadInfo(info);
      setStep(2);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setUploadError(msg);
      setStep(0);
      setFileName(null);
    } finally {
      setUploading(false);
    }
  }

  function resetUpload() {
    setFileName(null);
    setUploadInfo(null);
    setUploadError(null);
    setStep(0);
    setCurrentJob(null);
    if (pollRef.current) clearInterval(pollRef.current);
  }

  async function startIngest() {
    if (!uploadInfo) return;
    setStep(3);
    try {
      const { job_id } = await runIngestion(uploadInfo.filename);
      pollRef.current = setInterval(async () => {
        try {
          const job = await getIngestStatus(job_id);
          setCurrentJob(job);
          if (job.status === "done" || job.status === "error") {
            if (pollRef.current) clearInterval(pollRef.current);
            loadPastJobs();
            if (job.status === "done") setStep(4);
          }
        } catch {
          // keep polling
        }
      }, 1500);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setUploadError(msg);
      setStep(2);
    }
  }

  const running = currentJob?.status === "running" || currentJob?.status === "pending";
  const progress = currentJob?.progress_pct ?? 0;

  const steps = [
    { num: "01", name: "Upload", info: "CSV · ≤ 250 MB" },
    { num: "02", name: "Validate", info: "Column detection" },
    { num: "03", name: "Run batch", info: "Embed → ChromaDB" },
    { num: "04", name: "Done", info: "Index updated" },
  ];

  return (
    <>
      <div className="topbar">
        <div className="topbar-right">
          <span>
            {pastJobs.length > 0
              ? `${pastJobs.length} past run${pastJobs.length !== 1 ? "s" : ""}`
              : "No past runs"}
          </span>
          <button className="btn ghost sm" onClick={loadPastJobs} disabled={loadingPast}>
            <RefreshCw size={12} className={loadingPast ? "spin" : ""} /> Refresh
          </button>
          <button className="btn ghost sm">
            <Info size={12} /> Docs
          </button>
        </div>
      </div>

      <div className="page" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {/* Stepper */}
        <div className="stepper">
          {steps.map((s, i) => (
            <div
              key={s.num}
              className={`step${i < step ? " done" : ""}${i === step ? " active" : ""}`}
            >
              <div className="num">{s.num}</div>
              <div className="name">{s.name}</div>
              <div className="info">{s.info}</div>
              {/* only show checkmark on steps before the last one */}
              {i < step && i < steps.length - 1 && (
                <span style={{ position: "absolute", top: 14, right: 16, color: "#4e8f6a" }}>
                  <Check size={14} strokeWidth={2.2} />
                </span>
              )}
            </div>
          ))}
        </div>

        {/* Error banner */}
        {uploadError && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 14px",
              background: "#fff0f0",
              border: "1px solid #fca5a5",
              borderRadius: 8,
              color: "#b91c1c",
              fontSize: 13,
            }}
          >
            <AlertCircle size={16} />
            {uploadError}
            <button
              className="btn ghost sm"
              style={{ marginLeft: "auto" }}
              onClick={() => setUploadError(null)}
            >
              <X size={12} />
            </button>
          </div>
        )}

        {/* Top row: upload + run always side-by-side */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          {/* Left: upload card */}
          <div className="card" style={{ padding: 18 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 10,
              }}
            >
              <div>
                <span className="eyebrow">File source</span>
                <h2
                  className="display"
                  style={{ fontSize: 16, margin: "4px 0 0", fontWeight: 400 }}
                >
                  {fileName ?? "No file selected"}
                </h2>
              </div>
              {fileName && (
                <button className="btn ghost sm" onClick={resetUpload}>
                  <X size={12} /> Remove
                </button>
              )}
            </div>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            <div
              className="dropzone"
              style={{ padding: 18, cursor: uploading ? "wait" : "pointer" }}
              onClick={() => !uploading && !fileName && fileRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (!uploading && !fileName) {
                  const f = e.dataTransfer.files[0];
                  if (f) handleFile(f);
                }
              }}
            >
              <Upload size={20} strokeWidth={1.2} />
              <div className="display" style={{ fontSize: 14, fontWeight: 400 }}>
                {uploading ? "Uploading…" : "Drop a CSV here, or click to browse"}
              </div>
            </div>
          </div>

          {/* Right: run card — placeholder until file is ready */}
          <div
            className="card"
            style={{
              padding: 18,
              display: "flex",
              flexDirection: "column",
              gap: 12,
              opacity: uploadInfo ? 1 : 0.45,
              transition: "opacity 0.2s",
            }}
          >
            <div>
              <span className="eyebrow">Batch run</span>
              <h2
                className="display"
                style={{ fontSize: 16, margin: "4px 0 0", fontWeight: 400, color: uploadInfo ? undefined : "var(--ink-3)" }}
              >
                {uploadInfo ? uploadInfo.filename : "Add a source first"}
              </h2>
            </div>
            <ul
              style={{
                margin: 0,
                padding: 0,
                listStyle: "none",
                display: "flex",
                flexDirection: "column",
                gap: 7,
                fontSize: 12,
                color: "var(--ink-2)",
              }}
            >
              {(uploadInfo
                ? [
                    [`${uploadInfo.row_count.toLocaleString()} rows to process`, true],
                    ["Deduplicate by URL", true],
                    ["Decimal normalisation", true],
                    ["Embedding model · all-MiniLM-L6-v2", true],
                    ["Upsert into ChromaDB (incremental)", true],
                  ]
                : [
                    ["Rows to process", false],
                    ["Deduplicate by URL", false],
                    ["Decimal normalisation", false],
                    ["Embedding model · all-MiniLM-L6-v2", false],
                    ["Upsert into ChromaDB (incremental)", false],
                  ]
              ).map(([label, ok], i) => (
                <li key={i} style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>{label as string}</span>
                  <span style={{ color: ok ? "#4e8f6a" : "var(--ink-3)" }}>
                    {ok ? <Check size={13} strokeWidth={2.2} /> : "—"}
                  </span>
                </li>
              ))}
            </ul>

            {/* Progress bar */}
            {uploadInfo && (running || currentJob?.status === "done" || currentJob?.status === "error") && (
              <div>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: 6,
                    fontFamily: "var(--display)",
                    fontSize: 11,
                    letterSpacing: "0.1em",
                    color: currentJob?.status === "error" ? "#b91c1c" : "var(--ink-3)",
                  }}
                >
                  <span>
                    {currentJob?.status === "running"
                      ? progress < 40
                        ? "Running pipeline…"
                        : "Indexing…"
                      : currentJob?.status === "done"
                      ? "Done"
                      : currentJob?.status === "error"
                      ? `Error: ${currentJob.message ?? "unknown"}`
                      : "Starting…"}
                  </span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <div className="bar">
                  <span
                    style={{
                      width: `${progress}%`,
                      background: currentJob?.status === "error" ? "#ef4444" : undefined,
                    }}
                  />
                </div>
              </div>
            )}

            <div style={{ marginTop: "auto" }}>
              <button
                className="btn primary"
                style={{ width: "100%" }}
                onClick={startIngest}
                disabled={!uploadInfo || running || !uploadInfo.valid || step >= 3}
              >
                {running ? "Running…" : step === 4 ? "Done" : "Run ingestion"}
              </button>
            </div>
          </div>
        </div>

        {/* Validation chips — compact strip, only after upload */}
        {step >= 2 && uploadInfo && (
          <div className="card" style={{ padding: "10px 18px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <span className="eyebrow" style={{ margin: 0, flexShrink: 0 }}>
                {uploadInfo.detected_columns.length} cols · {uploadInfo.row_count.toLocaleString()} rows
              </span>
              <span className={`chip ${uploadInfo.valid ? "lav" : ""}`} style={{ flexShrink: 0 }}>
                {uploadInfo.valid ? "valid" : `${uploadInfo.missing_required.length} missing`}
              </span>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {uploadInfo.detected_columns.map((col) => (
                  <span
                    key={col}
                    style={{
                      padding: "2px 8px",
                      borderRadius: 4,
                      fontSize: 11,
                      fontFamily: "var(--display)",
                      background: uploadInfo.missing_required.includes(col) ? "#fef2f2" : "var(--grape-line)",
                      color: uploadInfo.missing_required.includes(col) ? "#b91c1c" : "var(--ink-2)",
                      border: "1px solid",
                      borderColor: uploadInfo.missing_required.includes(col) ? "#fca5a5" : "transparent",
                    }}
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Preview table — only after upload */}
        {step >= 2 && uploadInfo && (
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: "10px 18px",
                borderBottom: "1px solid var(--grape-line)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span className="eyebrow">Preview · first {uploadInfo.preview.length} rows</span>
              {uploadInfo.valid && <span className="chip lav">all rows valid</span>}
            </div>
            <div className="scroll-y" style={{ maxHeight: 200 }}>
              <table className="tbl">
                <thead>
                  <tr>
                    {uploadInfo.detected_columns.slice(0, 5).map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadInfo.preview.map((row, i) => (
                    <tr key={i}>
                      {uploadInfo.detected_columns.slice(0, 5).map((col) => (
                        <td
                          key={col}
                          style={{
                            maxWidth: 160,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {String(row[col] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Past runs */}
        {pastJobs.length > 0 && (
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                padding: "10px 18px",
                borderBottom: "1px solid var(--grape-line)",
              }}
            >
              <span className="eyebrow">Past ingestion runs</span>
            </div>
            <div className="scroll-y" style={{ maxHeight: 200 }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Status</th>
                    <th style={{ textAlign: "right" }}>Rows</th>
                    <th style={{ textAlign: "right" }}>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {pastJobs.map((j) => (
                    <tr key={j.id}>
                      <td style={{ fontWeight: 500 }}>{j.filename}</td>
                      <td>
                        <span
                          style={{
                            padding: "2px 8px",
                            borderRadius: 4,
                            fontSize: 11,
                            fontFamily: "var(--display)",
                            letterSpacing: "0.08em",
                            background:
                              j.status === "done"
                                ? "#ecfdf5"
                                : j.status === "error"
                                ? "#fef2f2"
                                : j.status === "running"
                                ? "#eff6ff"
                                : "var(--grape-line)",
                            color:
                              j.status === "done"
                                ? "#15803d"
                                : j.status === "error"
                                ? "#b91c1c"
                                : j.status === "running"
                                ? "#1d4ed8"
                                : "var(--ink-3)",
                          }}
                        >
                          {j.status}
                        </span>
                      </td>
                      <td style={{ textAlign: "right", fontFamily: "var(--display)" }}>
                        {j.total_rows?.toLocaleString() ?? "—"}
                      </td>
                      <td
                        style={{
                          textAlign: "right",
                          color: "var(--ink-3)",
                          fontSize: 12,
                        }}
                      >
                        {new Date(j.created_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
