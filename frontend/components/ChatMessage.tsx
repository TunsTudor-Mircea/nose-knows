"use client";

import { ThumbsDown, ThumbsUp } from "lucide-react";
import type { ChatMessage as ChatMessageType } from "@/lib/types";
import PerfumeCard from "./PerfumeCard";

interface Props {
  message: ChatMessageType;
  onFeedback: (messageId: string, score: 1 | -1) => void;
}

const INTENT_LABELS: Record<string, string> = {
  mood_based: "Mood",
  occasion_based: "Occasion",
  note_based: "Notes",
  follow_up: "Follow-up",
};

export default function ChatMessage({ message, onFeedback }: Props) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-indigo-600 text-white px-4 py-3 text-sm shadow-sm">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Intent badge */}
      {message.intent && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
            Intent
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-violet-100 text-violet-700 font-medium">
            {INTENT_LABELS[message.intent] ?? message.intent}
          </span>
        </div>
      )}

      {/* Agent bubble */}
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm bg-stone-100 text-gray-800 px-4 py-3 text-sm shadow-sm whitespace-pre-wrap leading-relaxed">
        {message.content}
      </div>

      {/* HyDE doc (collapsible) */}
      {message.hyde_doc && (
        <details className="max-w-[85%] text-xs">
          <summary className="cursor-pointer text-gray-400 hover:text-gray-600 select-none">
            HyDE document
          </summary>
          <pre className="mt-2 p-3 rounded-xl bg-amber-50 border border-amber-100 text-amber-900 whitespace-pre-wrap font-mono text-[11px]">
            {message.hyde_doc}
          </pre>
        </details>
      )}

      {/* Perfume cards */}
      {message.perfumes && message.perfumes.length > 0 && (
        <div className="flex gap-3 overflow-x-auto pb-1">
          {message.perfumes.map((p, i) => (
            <PerfumeCard key={`${p.perfume}-${i}`} perfume={p} />
          ))}
        </div>
      )}

      {/* Feedback buttons */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-400">Was this helpful?</span>
        <button
          onClick={() => onFeedback(message.id, 1)}
          disabled={message.feedback !== null && message.feedback !== undefined}
          aria-label="Thumbs up"
          className={`p-1.5 rounded-lg transition-colors ${
            message.feedback === 1
              ? "bg-green-100 text-green-600"
              : "text-gray-400 hover:bg-green-50 hover:text-green-600"
          } disabled:cursor-default`}
        >
          <ThumbsUp className="w-4 h-4" />
        </button>
        <button
          onClick={() => onFeedback(message.id, -1)}
          disabled={message.feedback !== null && message.feedback !== undefined}
          aria-label="Thumbs down"
          className={`p-1.5 rounded-lg transition-colors ${
            message.feedback === -1
              ? "bg-red-100 text-red-600"
              : "text-gray-400 hover:bg-red-50 hover:text-red-600"
          } disabled:cursor-default`}
        >
          <ThumbsDown className="w-4 h-4" />
        </button>
        {message.feedback !== null && message.feedback !== undefined && (
          <span className="text-xs text-gray-400">Thanks for the feedback!</span>
        )}
      </div>
    </div>
  );
}
