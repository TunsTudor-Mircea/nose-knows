"use client";

import { Settings2, X } from "lucide-react";
import type { Filters } from "@/lib/types";

interface Props {
  filters: Filters;
  onChange: (filters: Filters) => void;
  isOpen: boolean;
  onClose: () => void;
}

const GENDERS = ["", "women", "men", "unisex"] as const;
const COMMON_ACCORDS = [
  "",
  "floral",
  "woody",
  "musky",
  "citrus",
  "oriental",
  "fresh",
  "amber",
  "vanilla",
  "gourmand",
  "aquatic",
  "spicy",
];

export default function Sidebar({ filters, onChange, isOpen, onClose }: Props) {
  const update = (patch: Partial<Filters>) => onChange({ ...filters, ...patch });

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-20 md:hidden"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <aside
        className={`fixed top-0 right-0 h-full w-72 bg-white shadow-xl z-30 flex flex-col transition-transform duration-200
          ${isOpen ? "translate-x-0" : "translate-x-full"} md:relative md:translate-x-0 md:shadow-none md:border-l md:border-stone-200`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-stone-200">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-indigo-600" />
            <span className="font-semibold text-gray-800 text-sm">Search options</span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 md:hidden"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Controls */}
        <div className="flex-1 overflow-y-auto px-5 py-5 flex flex-col gap-6">
          {/* HyDE toggle */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-700">HyDE retrieval</p>
              <p className="text-xs text-gray-400 mt-0.5">
                Generate a hypothetical document before searching
              </p>
            </div>
            <button
              onClick={() => update({ use_hyde: !filters.use_hyde })}
              role="switch"
              aria-checked={filters.use_hyde}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors
                ${filters.use_hyde ? "bg-indigo-600" : "bg-gray-200"}`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow ring-0 transition-transform
                  ${filters.use_hyde ? "translate-x-5" : "translate-x-0"}`}
              />
            </button>
          </div>

          {/* Top-K slider */}
          <div>
            <label className="text-sm font-medium text-gray-700">
              Results: <span className="text-indigo-600 font-bold">{filters.top_k}</span>
            </label>
            <input
              type="range"
              min={1}
              max={10}
              value={filters.top_k}
              onChange={(e) => update({ top_k: Number(e.target.value) })}
              className="w-full mt-2 accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-0.5">
              <span>1</span>
              <span>10</span>
            </div>
          </div>

          {/* Gender filter */}
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">Gender</label>
            <div className="flex flex-wrap gap-2">
              {GENDERS.map((g) => (
                <button
                  key={g || "any"}
                  onClick={() => update({ gender: g })}
                  className={`text-xs px-3 py-1 rounded-full border transition-colors
                    ${filters.gender === g
                      ? "bg-indigo-600 text-white border-indigo-600"
                      : "bg-white text-gray-600 border-gray-300 hover:border-indigo-400"
                    }`}
                >
                  {g || "Any"}
                </button>
              ))}
            </div>
          </div>

          {/* Accord filter */}
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-2">Accord</label>
            <div className="flex flex-wrap gap-2">
              {COMMON_ACCORDS.map((a) => (
                <button
                  key={a || "any"}
                  onClick={() => update({ accord: a })}
                  className={`text-xs px-3 py-1 rounded-full border transition-colors
                    ${filters.accord === a
                      ? "bg-indigo-600 text-white border-indigo-600"
                      : "bg-white text-gray-600 border-gray-300 hover:border-indigo-400"
                    }`}
                >
                  {a || "Any"}
                </button>
              ))}
            </div>
          </div>

          {/* Brand filter */}
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">Brand</label>
            <input
              type="text"
              placeholder="e.g. Chanel, Dior…"
              value={filters.brand}
              onChange={(e) => update({ brand: e.target.value })}
              className="w-full text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-400"
            />
          </div>

          {/* Reset */}
          <button
            onClick={() =>
              onChange({ gender: "", accord: "", brand: "", top_k: 5, use_hyde: true })
            }
            className="text-xs text-gray-400 hover:text-red-500 transition-colors self-start"
          >
            Reset all filters
          </button>
        </div>
      </aside>
    </>
  );
}
