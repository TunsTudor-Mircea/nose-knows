export function slugToTitle(s: string): string {
  if (!s) return s;
  return s.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
