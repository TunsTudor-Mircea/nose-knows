const NOTE_CATEGORIES: [string, string[]][] = [
  ["citrus",  ["bergamot","lemon","mandarin","orange","yuzu","neroli","grapefruit","citrus","lime","petitgrain"]],
  ["fruit",   ["apple","raspberry","blackcurrant","cassis","pineapple","pear","fig","plum","peach","fruit","berry","cherry"]],
  ["floral",  ["rose","jasmine","ylang","violet","freesia","lily","blossom","iris","muguet","tuberose","orchid","floral","peony","magnolia"]],
  ["woody",   ["cedar","sandalwood","guaiac","oak","oakmoss","wood","papyrus","vetiver","cypress"]],
  ["spice",   ["pepper","clove","cinnamon","cardamom","anise","nutmeg","saffron","ginger","spic"]],
  ["sweet",   ["vanilla","tonka","honey","caramel","almond","coconut","cocoa","sweet","gourmand","praline"]],
  ["musk",    ["musk","ambergris","ambrox","amber","ambrette","suede","leather"]],
  ["green",   ["green","juniper","fig leaf","pine","tea","mint","basil","herb","mate","grass"]],
  ["aquatic", ["water","marine","salt","sea","ocean","aqua"]],
  ["smoky",   ["smok","incense","tobacco","birch","tar","ink"]],
];

export function noteCategory(name: string): string {
  const n = (name || "").toLowerCase();
  for (const [cat, kws] of NOTE_CATEGORIES) {
    for (const k of kws) if (n.includes(k)) return cat;
  }
  return "default";
}
