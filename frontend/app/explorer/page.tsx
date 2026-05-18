"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import { noteCategory } from "@/lib/notes";
import { listPerfumes } from "@/lib/api";
import type { PerfumeListItem } from "@/lib/types";

type SortKey = "rating" | "name";

function Notes({ list }: { list: string[] }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
      {list.map((n) => (
        <span key={n} className={`chip nc-${noteCategory(n)}`}>{n}</span>
      ))}
    </div>
  );
}

const PAGE_SIZE = 50;

export default function ExplorerPage() {
  const [rows, setRows] = useState<PerfumeListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState("");
  const [gender, setGender] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<SortKey>("rating");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  const load = useCallback(async (search: string, genderFilter: string, pageNum: number) => {
    setLoading(true);
    try {
      const res = await listPerfumes({ search, gender: genderFilter, limit: PAGE_SIZE, offset: pageNum * PAGE_SIZE });
      setTotal(res.total);
      setRows(res.perfumes);
      setSelectedId(res.perfumes.length > 0 ? res.perfumes[0].id : null);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load("", "", 0);
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const handleSearch = useCallback(() => {
    setPage(0);
    load(q, [...gender].join(","), 0);
  }, [q, gender, load]);

  // Re-run search whenever gender selection changes.
  useEffect(() => {
    setPage(0);
    load(q, [...gender].join(","), 0);
  }, [gender]);  // eslint-disable-line react-hooks/exhaustive-deps

  const totalPages = Math.ceil(total / PAGE_SIZE);

  function goToPage(newPage: number) {
    setPage(newPage);
    load(q, [...gender].join(","), newPage);
  }

  const sorted = useMemo(() => {
    const r = [...rows];
    if (sortBy === "rating") r.sort((a, b) => (b.rating ?? 0) - (a.rating ?? 0));
    else r.sort((a, b) => a.perfume.localeCompare(b.perfume));
    return r;
  }, [rows, sortBy]);

  const selected = sorted.find((p) => p.id === selectedId) ?? sorted[0] ?? null;

  function toggleGender(g: string) {
    setGender((prev) => {
      const next = new Set(prev);
      next.has(g) ? next.delete(g) : next.add(g);
      return next;
    });
  }

  return (
    <>
      <div className="page fit">
        <div style={{ display: "flex", gap: 12, marginBottom: 18 }}>
          <div className="search" style={{ flex: 1 }}>
            <Search size={15} />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search perfume name or brand..."
            />
            <kbd className="kbd">/</kbd>
          </div>
          <select
            className="select"
            style={{ width: 220 }}
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortKey)}
          >
            <option value="rating">Sort · rating high to low</option>
            <option value="name">Sort · name A to Z</option>
          </select>
          <button className="btn primary sm" onClick={handleSearch}>Search</button>
        </div>

        <div className="explorer-grid">
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "12px 18px", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--grape-line)" }}>
              <span className="eyebrow">
                {loading ? "Loading…" : `${total.toLocaleString()} matches · page ${page + 1} of ${totalPages || 1}`}
              </span>
              <div style={{ display: "flex", gap: 6 }}>
                {[...gender].map((g) => (
                  <span key={g} className="chip" style={{ cursor: "pointer" }} onClick={() => toggleGender(g)}>
                    {g} <X size={10} />
                  </span>
                ))}
              </div>
            </div>
            <div className="scroll-y" style={{ flex: 1, minHeight: 0 }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: 40 }}></th>
                    <th>Perfume</th>
                    <th>Brand</th>
                    <th>Gender</th>
                    <th>Accords</th>
                    <th style={{ width: 90, textAlign: "right" }}>Rating</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((p, i) => (
                    <tr
                      key={p.id}
                      className={selectedId === p.id ? "selected" : ""}
                      onClick={() => setSelectedId(p.id)}
                      style={{ cursor: "pointer" }}
                    >
                      <td style={{ color: "var(--ink-3)", fontFamily: "var(--display)", fontSize: 11 }}>
                        {String(page * PAGE_SIZE + i + 1).padStart(3, "0")}
                      </td>
                      <td style={{ fontWeight: 500 }}>{p.perfume}</td>
                      <td style={{ color: "var(--ink-2)" }}>{p.brand}</td>
                      <td>
                        {p.gender && (
                          <span className="chip lav" style={{ padding: "2px 8px", fontSize: 10 }}>{p.gender}</span>
                        )}
                      </td>
                      <td>
                        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                          {p.accords.slice(0, 3).map((a) => (
                            <span key={a} className="chip" style={{ padding: "2px 8px", fontSize: 10 }}>{a}</span>
                          ))}
                        </div>
                      </td>
                      <td style={{ textAlign: "right", fontFamily: "var(--display)", fontWeight: 500 }}>
                        {p.rating != null ? p.rating.toFixed(2) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {totalPages > 1 && (
              <div style={{ padding: "10px 18px", display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid var(--grape-line)" }}>
                <button
                  className="btn sm"
                  disabled={page === 0 || loading}
                  onClick={() => goToPage(page - 1)}
                >
                  ← Prev
                </button>
                <span className="eyebrow">
                  {page + 1} / {totalPages}
                </span>
                <button
                  className="btn sm"
                  disabled={page >= totalPages - 1 || loading}
                  onClick={() => goToPage(page + 1)}
                >
                  Next →
                </button>
              </div>
            )}
          </div>

          <div className="facet-rail">
            {selected && (
              <div className="card" style={{ padding: 22 }}>
                <span className="eyebrow">
                  Detail · row {String(page * PAGE_SIZE + sorted.findIndex((r) => r.id === selected.id) + 1).padStart(3, "0")}
                </span>
                <h2 className="display" style={{ fontSize: 22, margin: "10px 0 4px", fontWeight: 400 }}>
                  {selected.perfume}
                </h2>
                <div style={{ fontFamily: "var(--display)", fontSize: 11.5, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--ink-3)" }}>
                  {selected.brand}
                </div>
                <div style={{ display: "flex", gap: 16, marginTop: 14, paddingBottom: 14, borderBottom: "1px solid var(--grape-line)" }}>
                  <div>
                    <div className="eyebrow">Rating</div>
                    <div style={{ fontFamily: "var(--display)", fontWeight: 200, fontSize: 28, marginTop: 4 }}>
                      {selected.rating != null ? selected.rating.toFixed(2) : "—"}
                    </div>
                  </div>
                  {selected.rating_count != null && (
                    <div>
                      <div className="eyebrow">Ratings</div>
                      <div style={{ fontFamily: "var(--display)", fontWeight: 200, fontSize: 28, marginTop: 4 }}>
                        {selected.rating_count > 999 ? `${(selected.rating_count / 1000).toFixed(1)}K` : selected.rating_count}
                      </div>
                    </div>
                  )}
                  {selected.year && (
                    <div>
                      <div className="eyebrow">Year</div>
                      <div style={{ fontFamily: "var(--display)", fontWeight: 200, fontSize: 28, marginTop: 4 }}>
                        {selected.year}
                      </div>
                    </div>
                  )}
                </div>
                <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                  {([["Top", selected.top_notes], ["Heart", selected.heart_notes], ["Base", selected.base_notes]] as [string, string[]][]).map(([lbl, ns]) => (
                    <div key={lbl}>
                      <div className="eyebrow" style={{ marginBottom: 6 }}>{lbl} notes</div>
                      <Notes list={ns} />
                    </div>
                  ))}
                  <div>
                    <div className="eyebrow" style={{ marginBottom: 6 }}>Accords</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                      {selected.accords.map((a) => (
                        <span key={a} className="chip lav">{a}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="facet">
              <h4>Gender</h4>
              {["women", "men", "unisex"].map((g) => (
                <div key={g} className="checkrow" onClick={() => toggleGender(g)}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                    <span className={`checkbox${gender.has(g) ? " on" : ""}`} />
                    {g}
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
