"""Pack every file of a results folder into one self-contained HTML page.

    python scripts/bundle_outputs.py outputs/final-run-20261022 --output outputs/final-run-20261022.html

Each file is embedded gzip-compressed (images as-is) with its size and SHA-256. Summaries,
logs and JSON open as text, figures as images and CSV tables as a preview of their first
rows. Every file, checkpoints included, downloads byte for byte from the page. An optional
--notes Markdown file adds an introduction (text before the first '## folder' heading) and
a short note above each folder's files.
"""

import argparse
import base64
from datetime import UTC, datetime
import gzip
from hashlib import sha256
import html
import json
from pathlib import Path
import re

TEXT = {".csv", ".json", ".log", ".md", ".txt", ".py", ".html", ".toml", ".tsv"}
IMAGES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}


def kind(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "table"
    if suffix in IMAGES:
        return "image"
    if suffix in TEXT:
        return "text"
    return "binary"


def entries(root, skip=()):
    for index, path in enumerate(sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() not in skip)):
        data = path.read_bytes()
        file_kind = kind(path)
        packed = data if file_kind == "image" else gzip.compress(data, 9, mtime=0)
        yield {"id": index, "path": path.relative_to(root).as_posix(), "kind": file_kind,
               "size": len(data), "sha256": sha256(data).hexdigest(), "gzip": file_kind != "image",
               "mime": IMAGES.get(path.suffix.lower(), "application/octet-stream"),
               "payload": base64.b64encode(packed).decode("ascii")}


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def markdown(text):
    """Paragraphs, bullet lists, bold and code: enough for short notes."""
    blocks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if all(line.lstrip().startswith("- ") for line in lines):
            blocks.append("<ul>" + "".join(f"<li>{inline(line.lstrip()[2:])}</li>" for line in lines) + "</ul>")
        elif lines and lines[0].startswith("### "):
            blocks.append(f"<h3>{inline(lines[0][4:])}</h3>" + (f"<p>{inline(' '.join(lines[1:]))}</p>" if lines[1:] else ""))
        else:
            blocks.append(f"<p>{inline(' '.join(lines))}</p>")
    return "".join(blocks)


def read_notes(path):
    """Text before the first '## folder' heading introduces the page; each section describes a folder."""
    if path is None:
        return "", {}
    intro, *sections = re.split(r"(?m)^## ", Path(path).read_text(encoding="utf-8"))
    notes = {}
    for section in sections:
        heading, _, body = section.partition("\n")
        notes[heading.strip()] = markdown(body)
    return markdown(intro), notes


def overview(root):
    """Static run summary from the pipeline's own records, when they exist."""
    parts = []
    completion = root / "completion.json"
    if completion.exists():
        record = json.loads(completion.read_text(encoding="utf-8"))
        counts = record.get("effective_stage_counts", {})
        integrity = record.get("file_integrity", {})
        devices = record.get("training_device_checks", [])
        parts.append(
            f"<p><b>{'Complete' if record.get('complete') else 'Not complete'}</b> · "
            + " · ".join(f"{html.escape(k.replace('_', ' '))}: {v}" for k, v in counts.items())
            + f" · hashes checked: {integrity.get('files', 0)} files, {len(integrity.get('errors', []))} errors"
            + f" · GPU checks passed: {sum(d.get('passed', False) for d in devices)}/{len(devices)}</p>")
    status = root / "run-status.json"
    if status.exists():
        stages = json.loads(status.read_text(encoding="utf-8")).get("stages", [])
        final = json.loads(completion.read_text(encoding="utf-8")).get("stages", {}) if completion.exists() else {}
        rows = "".join(
            f"<tr><td>{html.escape(s['name'])}</td><td>{html.escape(final.get(s['name'], s.get('state', '')))}</td>"
            f"<td class=num>{s.get('elapsed_seconds', '')}</td></tr>" for s in stages)
        parts.append("<details><summary>Stages</summary><table><tr><th>Stage</th><th>Result</th>"
                     f"<th class=num>Seconds</th></tr>{rows}</table></details>")
    return "".join(parts)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#1d2430;--muted:#5d6878;--line:#d9dee5;--panel:#f4f6f9;--accent:#086a82}
@media (prefers-color-scheme:dark){:root{--bg:#14181e;--fg:#e4e8ee;--muted:#9aa5b4;--line:#2c333d;--panel:#1c2229;--accent:#5fb8cf}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}h2{font-size:16px;margin:24px 0 8px;border-bottom:1px solid var(--line)}
.muted{color:var(--muted)}.note{max-width:820px}.note p,.note li{margin:4px 0}h3{font-size:15px;margin:16px 0 4px}code{font:12px ui-monospace,Consolas,monospace}input{width:100%;box-sizing:border-box;padding:8px;font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px}
details{border:1px solid var(--line);border-radius:6px;margin:6px 0;background:var(--panel)}
summary{cursor:pointer;padding:6px 10px;display:flex;gap:12px;flex-wrap:wrap}summary b{word-break:break-all}
.body{padding:8px 10px;overflow:auto;max-height:70vh;background:var(--bg)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.4 ui-monospace,Consolas,monospace}
table{border-collapse:collapse;font:12px ui-monospace,Consolas,monospace}td,th{border:1px solid var(--line);padding:2px 6px;white-space:nowrap}
.num{text-align:right}img{max-width:100%;background:#fff}button{font:inherit;cursor:pointer;margin:4px 0;color:var(--accent);background:none;border:1px solid var(--accent);border-radius:4px}
</style></head><body><main>
<h1>__TITLE__</h1>
<p class=muted>__COUNT__ files, __SIZE__ in total, packed __WHEN__. Open a file to view it; every file downloads byte for byte.</p>
<div class=note>__INTRO__</div>
__OVERVIEW__
<h2>Files</h2>
<input id=filter placeholder="Filter by path, e.g. performance.csv or matched-risklab/">
<div id=list></div>
</main>
<script id=manifest type="application/json">__MANIFEST__</script>
<script id=notes type="application/json">__NOTES__</script>
__PAYLOADS__
<script>
const files = JSON.parse(document.getElementById("manifest").textContent);
const notes = JSON.parse(document.getElementById("notes").textContent);
const fmt = n => n < 1024 ? n + " B" : n < 1048576 ? (n / 1024).toFixed(1) + " KB" : (n / 1048576).toFixed(1) + " MB";
async function bytes(f) {
  const raw = Uint8Array.from(atob(document.getElementById("f" + f.id).textContent), c => c.charCodeAt(0));
  if (!f.gzip) return raw;
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
function parseCsv(text, limit) {
  const rows = []; let row = [], cell = "", quoted = false, i = 0;
  for (; i < text.length && rows.length <= limit; i++) {
    const c = text[i];
    if (quoted) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false; } else cell += c; }
    else if (c === '"') quoted = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (c !== "\\r") cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}
async function render(f, box) {
  const data = await bytes(f);
  const save = document.createElement("button");
  save.textContent = "Download " + f.path.split("/").pop();
  save.onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([data], {type: f.mime}));
    a.download = f.path.split("/").pop(); a.click(); };
  box.append(save);
  if (f.kind === "image") { const img = document.createElement("img"); img.src = URL.createObjectURL(new Blob([data], {type: f.mime})); box.append(img); return; }
  if (f.kind === "binary") { box.append(document.createTextNode(" Binary file; download to open it.")); return; }
  const text = new TextDecoder().decode(data);
  if (f.kind === "table") {
    const limit = 300, rows = parseCsv(text, limit), total = text.split("\\n").filter(Boolean).length - 1;
    const note = document.createElement("p"); note.className = "muted";
    note.textContent = total > limit ? `Showing the first ${limit} of ${total} rows; download for all of them.` : `${total} rows.`;
    const table = document.createElement("table");
    rows.slice(0, limit + 1).forEach((r, n) => { const tr = table.insertRow();
      r.forEach(v => { const td = document.createElement(n ? "td" : "th"); td.textContent = v;
        if (n && v !== "" && !isNaN(v)) td.className = "num"; tr.append(td); }); });
    box.append(note, table); return;
  }
  const pre = document.createElement("pre");
  pre.textContent = f.path.endsWith(".json") ? (() => { try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; } })() : text;
  box.append(pre);
}
const list = document.getElementById("list");
let folder = null;
for (const f of files) {
  const top = f.path.includes("/") ? f.path.split("/")[0] : "(top level)";
  if (top !== folder) { folder = top; const h = document.createElement("h2"); h.textContent = top; h.dataset.folder = top; list.append(h);
    if (notes[top]) { const n = document.createElement("div"); n.className = "note"; n.dataset.folder = top; n.innerHTML = notes[top]; list.append(n); } }
  const d = document.createElement("details"); d.dataset.path = f.path.toLowerCase(); d.dataset.folder = top;
  d.innerHTML = `<summary><b></b><span class=muted>${fmt(f.size)} · ${f.kind} · sha256 ${f.sha256.slice(0, 16)}</span></summary><div class=body></div>`;
  d.querySelector("b").textContent = f.path;
  d.addEventListener("toggle", () => { const box = d.querySelector(".body"); if (d.open && !box.childElementCount) render(f, box); });
  list.append(d);
}
document.getElementById("filter").addEventListener("input", e => {
  const q = e.target.value.toLowerCase(), shown = new Set();
  list.querySelectorAll("details").forEach(d => { const ok = d.dataset.path.includes(q); d.hidden = !ok; if (ok) shown.add(d.dataset.folder); });
  list.querySelectorAll("h2, div.note").forEach(h => { h.hidden = !shown.has(h.dataset.folder); });
});
</script></body></html>
"""


def bundle(root, output, title=None, notes=None):
    root, output = Path(root), Path(output)
    intro, folder_notes = read_notes(notes)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    items = list(entries(root, skip={output.resolve()}))
    manifest = [{k: v for k, v in item.items() if k != "payload"} for item in items]
    payloads = "\n".join(f'<script id="f{item["id"]}" type="application/octet-stream">{item["payload"]}</script>'
                         for item in items)
    total = sum(item["size"] for item in items)
    page = (PAGE.replace("__TITLE__", html.escape(title or f"All outputs: {root.name}"))
            .replace("__COUNT__", f"{len(items):,}")
            .replace("__SIZE__", f"{total / 1048576:,.1f} MB")
            .replace("__WHEN__", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
            .replace("__INTRO__", intro)
            .replace("__OVERVIEW__", overview(root))
            .replace("__NOTES__", json.dumps(folder_notes).replace("</", "<\\/"))
            .replace("__MANIFEST__", json.dumps(manifest).replace("</", "<\\/"))
            .replace("__PAYLOADS__", payloads))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(page)
    return len(items), total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="results folder to pack, e.g. outputs/final-run-20261022")
    parser.add_argument("--output", type=Path, required=True, help="new HTML file; never overwritten")
    parser.add_argument("--title", help="page heading (default: the folder name)")
    parser.add_argument("--notes", type=Path, help="Markdown introduction and per-folder notes")
    args = parser.parse_args(argv)
    count, total = bundle(args.folder, args.output, args.title, args.notes)
    print(f"Packed {count} files ({total / 1048576:.1f} MB) into {args.output} "
          f"({args.output.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
