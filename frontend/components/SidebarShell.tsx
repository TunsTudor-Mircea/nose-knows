"use client";
import { useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  MessageSquare, Database, Workflow, FlaskConical,
  ThumbsUp, ChevronLeft,
} from "lucide-react";

const NAV = [
  { href: "/chat",     label: "Consult",          Icon: MessageSquare },
  { href: "/explorer", label: "Dataset Explorer",  Icon: Database },
  { href: "/ingest",   label: "Ingestion",         Icon: Workflow },
  { href: "/eval",     label: "Evaluation Lab",    Icon: FlaskConical },
  { href: "/feedback", label: "Feedback",          Icon: ThumbsUp },
];

export default function SidebarShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();
  const router = useRouter();

  return (
    <div className={`shell${collapsed ? " collapsed" : ""}`}>
      <aside className="rail">
        <button className="rail-toggle" onClick={() => setCollapsed(c => !c)} aria-label="toggle sidebar">
          <ChevronLeft size={14} strokeWidth={1.8} />
        </button>

        <div className="brand">
          <div className="brand-mark">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logo.png" alt="NoseKnows" className="brand-img" />
          </div>
          <div className="brand-text">
            <span className="name">NoseKnows</span>
          </div>
        </div>

        <nav className="nav-list">
          {NAV.map(({ href, label, Icon }) => (
            <button
              key={href}
              className={`nav-item${pathname?.startsWith(href) ? " active" : ""}`}
              onClick={() => router.push(href)}
              title={label}
            >
              <Icon size={17} strokeWidth={1.5} />
              <span className="label">{label}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="workspace">{children}</main>
    </div>
  );
}
