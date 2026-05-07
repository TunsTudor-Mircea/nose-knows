"use client";

import type { PerfumeCard as PerfumeCardType } from "@/lib/types";
import { Star } from "lucide-react";

interface Props {
  perfume: PerfumeCardType;
}

const NOTE_LAYER_COLORS: Record<string, string> = {
  top: "bg-amber-100 text-amber-800",
  heart: "bg-rose-100 text-rose-800",
  base: "bg-stone-200 text-stone-800",
};

function NoteChips({
  label,
  notes,
  layer,
}: {
  label: string;
  notes: string[];
  layer: keyof typeof NOTE_LAYER_COLORS;
}) {
  if (!notes.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide w-14 shrink-0">
        {label}
      </span>
      {notes.map((note) => (
        <span
          key={note}
          className={`text-xs px-2 py-0.5 rounded-full font-medium ${NOTE_LAYER_COLORS[layer]}`}
        >
          {note}
        </span>
      ))}
    </div>
  );
}

export default function PerfumeCard({ perfume }: Props) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white shadow-sm p-4 flex flex-col gap-3 min-w-[220px] max-w-[280px]">
      {/* Header */}
      <div>
        <h3 className="font-bold text-gray-900 text-sm leading-tight">
          {perfume.perfume || "Unknown Perfume"}
        </h3>
        <p className="text-xs text-gray-500 mt-0.5">{perfume.brand || "Unknown Brand"}</p>
      </div>

      {/* Rating */}
      {perfume.rating != null && (
        <div className="flex items-center gap-1">
          <Star className="w-3.5 h-3.5 text-amber-400 fill-amber-400" />
          <span className="text-xs font-medium text-gray-700">
            {perfume.rating.toFixed(2)}
          </span>
        </div>
      )}

      {/* Notes */}
      <div className="flex flex-col gap-1.5">
        <NoteChips label="Top" notes={perfume.top_notes} layer="top" />
        <NoteChips label="Heart" notes={perfume.heart_notes} layer="heart" />
        <NoteChips label="Base" notes={perfume.base_notes} layer="base" />
      </div>

      {/* Accords */}
      {perfume.accords.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {perfume.accords.map((accord) => (
            <span
              key={accord}
              className="text-xs px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 font-medium border border-indigo-100"
            >
              {accord}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
