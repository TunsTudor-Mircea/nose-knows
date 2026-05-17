"use client";
import { usePathname, useRouter } from "next/navigation";
import {
  MessageSquare, Database, Workflow, FlaskConical, ThumbsUp,
} from "lucide-react";

const NAV = [
  { href: "/chat",     label: "Consult"},
  { href: "/explorer", label: "Dataset Explorer"},
  { href: "/ingest",   label: "Ingestion"},
  { href: "/eval",     label: "Evaluation Lab"},
  { href: "/feedback", label: "Feedback"},
];

export default function HeaderShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <div className="shell">
      <header className="app-header">
        <div className="brand" onClick={() => router.push("/chat")} role="button">
          <div className="brand-mark">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo.png" alt="NoseKnows" className="brand-img" />
          </div>
          <div className="brand-text">
            <span className="name">NoseKnows</span>
          </div>
        </div>

        <nav className="nav-list">
          {NAV.map(({ href, label }) => (
            <button
              key={href}
              className={`nav-item${pathname?.startsWith(href) ? " active" : ""}`}
              onClick={() => router.push(href)}
              title={label}
            >
              <span className="label">{label}</span>
            </button>
          ))}
        </nav>
      </header>

      <main className="workspace">{children}</main>
    </div>
  );
}
