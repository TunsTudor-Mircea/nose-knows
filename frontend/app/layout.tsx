import type { Metadata } from "next";
import SidebarShell from "@/components/SidebarShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "NoseKnows",
  description: "Describe your mood, occasion, or desired feeling and NoseKnows will recommend the perfect fragrance.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <SidebarShell>{children}</SidebarShell>
      </body>
    </html>
  );
}
