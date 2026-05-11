import { noteCategory } from "@/lib/notes";
import type { PerfumeCard as PerfumeCardType } from "@/lib/types";

interface Props {
  card: PerfumeCardType;
  index: number;
}

function Notes({ list }: { list: string[] }) {
  return (
    <div className="notes">
      {list.map((name) => (
        <span key={name} className={`n nc-${noteCategory(name)}`}>{name}</span>
      ))}
    </div>
  );
}

export default function PerfumeCard({ card, index }: Props) {
  return (
    <div className="pcard">
      <div className="num">{String(index).padStart(2, "0")}</div>
      <div>
        <div className="title">{card.perfume}</div>
        <div className="brand-name">{card.brand}</div>
        <div className="pyramid">
          <div className="lbl">Top</div>    <Notes list={card.top_notes} />
          <div className="lbl">Heart</div>  <Notes list={card.heart_notes} />
          <div className="lbl">Base</div>   <Notes list={card.base_notes} />
        </div>
      </div>
      <div className="right">
        {card.rating != null && (
          <div className="rating-val">{card.rating.toFixed(2)}</div>
        )}
        <div className="accords">
          {card.accords.slice(0, 3).map((a) => (
            <span key={a} className="chip lav">{a}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
