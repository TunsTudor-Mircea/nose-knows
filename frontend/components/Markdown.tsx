interface Props {
  children: string;
  className?: string;
}

function inlineFormat(s: string): string {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}

export default function Markdown({ children, className }: Props) {
  const escaped = children
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  const html = escaped
    .split(/\n{2,}/)
    .map((block) => {
      const lines = block.trim().split("\n");
      if (!lines.length) return "";

      // Process line-by-line so mixed blocks (prose + bullets) render correctly.
      const parts: string[] = [];
      let ulItems: string[] = [];
      let olItems: string[] = [];

      const flushUl = () => {
        if (ulItems.length) {
          parts.push(`<ul>${ulItems.map((i) => `<li>${inlineFormat(i)}</li>`).join("")}</ul>`);
          ulItems = [];
        }
      };
      const flushOl = () => {
        if (olItems.length) {
          parts.push(`<ol>${olItems.map((i) => `<li>${inlineFormat(i)}</li>`).join("")}</ol>`);
          olItems = [];
        }
      };
      const paraLines: string[] = [];
      const flushPara = () => {
        if (paraLines.length) {
          parts.push(`<p>${paraLines.map(inlineFormat).join("<br/>")}</p>`);
          paraLines.length = 0;
        }
      };

      for (const line of lines) {
        if (/^[-*]\s/.test(line)) {
          flushPara();
          flushOl();
          ulItems.push(line.replace(/^[-*]\s+/, ""));
        } else if (/^\d+\.\s/.test(line)) {
          flushPara();
          flushUl();
          olItems.push(line.replace(/^\d+\.\s+/, ""));
        } else {
          flushUl();
          flushOl();
          paraLines.push(line);
        }
      }
      flushPara();
      flushUl();
      flushOl();

      return parts.join("");
    })
    .join("");

  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
