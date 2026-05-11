interface Props {
  children: string;
  className?: string;
}

export default function Markdown({ children, className }: Props) {
  const html = children
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      const lines = trimmed.split("\n");
      if (lines.every((l) => /^[-*]\s/.test(l))) {
        const items = lines.map((l) => `<li>${l.replace(/^[-*]\s/, "")}</li>`).join("");
        return `<ul>${items}</ul>`;
      }
      if (lines.every((l) => /^\d+\.\s/.test(l))) {
        const items = lines.map((l) => `<li>${l.replace(/^\d+\.\s/, "")}</li>`).join("");
        return `<ol>${items}</ol>`;
      }
      return `<p>${lines.join("<br/>")}</p>`;
    })
    .join("");

  return (
    <div
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
