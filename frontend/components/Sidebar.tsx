"use client";

import { Settings2, X } from "lucide-react";
import type { Filters } from "@/lib/types";

interface Props {
  filters: Filters;
  onChange: (filters: Filters) => void;
  isOpen: boolean;
  onClose: () => void;
}

export default function Sidebar({ filters, onChange, isOpen, onClose }: Props) {
  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-20 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed top-0 right-0 h-full w-72 bg-white shadow-xl z-30 flex flex-col transition-transform duration-200
          ${isOpen ? "translate-x-0" : "translate-x-full"} md:relative md:translate-x-0 md:shadow-none md:border-l md:border-stone-200`}
      >
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

        <div className="flex-1 overflow-y-auto px-5 py-5 flex flex-col gap-6">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Results: <span className="text-indigo-600 font-bold">{filters.top_k}</span>
            </label>
            <input
              type="range"
              min={1}
              max={20}
              value={filters.top_k}
              onChange={(e) => onChange({ top_k: Number(e.target.value) })}
              className="w-full mt-2 accent-indigo-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-0.5">
              <span>1</span>
              <span>20</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
