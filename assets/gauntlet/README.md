# Sir Leaks-a-Lot character art (optional)

The guardian avatar defaults to a built-in **aging wizard drawn in inline SVG**
(self-contained, no external requests). To use your own artwork instead, drop
image files here — they're embedded as data URIs, so nothing loads externally.

Naming (checked in this order per level):

- `<level>.png` — a distinct image per level, e.g. `1.png`, `2.png`, … `12.png`
  (also accepts `.jpg`, `.jpeg`, `.webp`, `.gif`, `.svg`).
- `character.png` — a single image used for every level.

Recommendations:
- Roughly square, ~150×150 or larger (it's shown at 150×150, cropped to fill).
- Use **royalty-free / your own** art. Do not copy another product's character
  (e.g. Lakera Gandalf's wizard) — that's their asset.

If no image is found, the built-in SVG wizard is used (it ages young→elder as the
levels get harder).
