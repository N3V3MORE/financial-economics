import base64
import gzip
import importlib.util
import json
from pathlib import Path
import re

import pytest


script_path = Path(__file__).resolve().parents[1] / "scripts" / "bundle_outputs.py"
spec = importlib.util.spec_from_file_location("bundle_outputs", script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def unpack(page):
    manifest = json.loads(re.search(r'<script id=manifest type="application/json">(.*?)</script>', page, re.S).group(1))
    files = {}
    for item in manifest:
        payload = re.search(rf'<script id="f{item["id"]}" type="application/octet-stream">(.*?)</script>', page).group(1)
        data = base64.b64decode(payload)
        files[item["path"]] = (item, gzip.decompress(data) if item["gzip"] else data)
    return files


def test_every_file_round_trips_byte_for_byte(tmp_path):
    root = tmp_path / "run"
    (root / "study" / "checkpoints").mkdir(parents=True)
    originals = {
        "completion.json": b'{"complete": true, "effective_stage_counts": {"passed": 2}}',
        "study/performance.csv": b'portfolio,sharpe\nSPY,0.79\n"a, quoted",1e-3\n',
        "study/figure.png": b"\x89PNG\r\n\x1a\nnot really an image",
        "study/checkpoints/seed_11.pt": bytes(range(256)) * 4,
        "logs/01-stage.log": "done – ok\n".encode(),
    }
    for name, data in originals.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    output = root / "bundle.html"
    count, total = module.bundle(root, output)
    files = unpack(output.read_text(encoding="utf-8"))
    assert count == len(originals) and total == sum(map(len, originals.values()))
    assert set(files) == set(originals)
    for name, data in originals.items():
        item, unpacked = files[name]
        assert unpacked == data
        assert item["size"] == len(data)
    assert files["study/performance.csv"][0]["kind"] == "table"
    assert files["study/figure.png"][0]["kind"] == "image"
    assert files["study/checkpoints/seed_11.pt"][0]["kind"] == "binary"
    assert "Complete" in output.read_text(encoding="utf-8")


def test_notes_add_an_introduction_and_folder_notes(tmp_path):
    root = tmp_path / "run"
    (root / "study").mkdir(parents=True)
    (root / "study" / "a.csv").write_text("x\n1\n")
    notes = tmp_path / "notes.md"
    notes.write_text("Why this run exists.\n\n## study\n**Decision:** use `cash`.\n\n- first\n- second <b>\n")
    output = tmp_path / "out.html"
    module.bundle(root, output, notes=notes)
    page = output.read_text(encoding="utf-8")
    assert "<p>Why this run exists.</p>" in page
    folder_notes = json.loads(re.search(r'<script id=notes type="application/json">(.*?)</script>', page, re.S)
                              .group(1).replace("<\\/", "</"))
    assert folder_notes["study"] == ("<p><b>Decision:</b> use <code>cash</code>.</p>"
                                     "<ul><li>first</li><li>second &lt;b&gt;</li></ul>")


def test_existing_output_is_never_replaced(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    output = tmp_path / "out.html"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        module.bundle(tmp_path, output)
    assert output.read_text() == "keep"
