"use client";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { searchBrands } from "@/lib/api";

interface Props {
  value: string;
  onChange: (brand: string) => void;
}

interface Rect { top: number; left: number; width: number; }

export default function BrandCombobox({ value, onChange }: Props) {
  const [query, setQuery] = useState(value);
  const [options, setOptions] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(0);
  const [rect, setRect] = useState<Rect | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync external clear (parent resets value to "").
  useEffect(() => { if (!value) setQuery(""); }, [value]);

  // Capture input position whenever dropdown opens or window resizes/scrolls.
  useLayoutEffect(() => {
    if (!open || !inputRef.current) return;
    const update = () => {
      if (!inputRef.current) return;
      const r = inputRef.current.getBoundingClientRect();
      setRect({ top: r.bottom + 4, left: r.left, width: r.width });
    };
    update();
    window.addEventListener("scroll", update, true);
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update, true);
      window.removeEventListener("resize", update);
    };
  }, [open]);

  // Fetch suggestions with debounce.
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!query.trim()) { setOptions([]); setOpen(false); return; }
    debounceRef.current = setTimeout(async () => {
      try {
        const results = await searchBrands(query);
        setOptions(results);
        setHighlighted(0);
        setOpen(results.length > 0);
      } catch { setOptions([]); }
    }, 220);
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, [query]);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as Node;
      if (inputRef.current && !inputRef.current.closest(".brand-combobox")?.contains(target)) {
        const dropdown = document.getElementById("brand-dropdown-portal");
        if (!dropdown?.contains(target)) setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const select = (brand: string) => { setQuery(brand); onChange(brand); setOpen(false); };
  const clear = () => { setQuery(""); onChange(""); setOpen(false); setOptions([]); };

  const handleKey = (e: React.KeyboardEvent) => {
    if (!open) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setHighlighted((h) => Math.min(h + 1, options.length - 1)); }
    if (e.key === "ArrowUp")   { e.preventDefault(); setHighlighted((h) => Math.max(h - 1, 0)); }
    if (e.key === "Enter")     { e.preventDefault(); if (options[highlighted]) select(options[highlighted]); }
    if (e.key === "Escape")    { setOpen(false); }
  };

  const dropdown = open && rect ? createPortal(
    <ul
      id="brand-dropdown-portal"
      className="brand-dropdown"
      style={{ top: rect.top, left: rect.left, width: rect.width }}
    >
      {options.map((b, i) => (
        <li
          key={b}
          className={i === highlighted ? "hi" : ""}
          onMouseDown={(e) => { e.preventDefault(); select(b); }}
          onMouseEnter={() => setHighlighted(i)}
        >
          {b}
        </li>
      ))}
    </ul>,
    document.body,
  ) : null;

  return (
    <div className="brand-combobox">
      <div className="brand-input-wrap">
        <input
          ref={inputRef}
          className="input"
          placeholder="Type a brand…"
          value={query}
          onChange={(e) => { setQuery(e.target.value); if (!e.target.value) onChange(""); }}
          onFocus={() => { if (options.length > 0) setOpen(true); }}
          onKeyDown={handleKey}
          autoComplete="off"
        />
        {query && (
          <button className="brand-clear" onClick={clear} aria-label="Clear brand">×</button>
        )}
      </div>
      {dropdown}
    </div>
  );
}
