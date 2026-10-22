"""Map provenance names to module files and recognize one reviewed relocation.

Run records name modules by their original flat filenames (``paper_gru.py``);
the manifest maps each to its grouped file and records the byte hashes from just
before and after the reviewed move. A historical single-module digest from before
the move is accepted while that module still matches its post-move hash.
"""

from hashlib import sha256
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
RELOCATION_MANIFEST = PACKAGE_ROOT / "_relocation.json"


def _manifest():
    return json.loads(RELOCATION_MANIFEST.read_text(encoding="utf-8"))


def module_source(name):
    """Find the implementation for an original logical module filename."""
    if Path(name).name != name or not name.endswith(".py"):
        raise ValueError("Expected an original module filename.")
    relative = _manifest()["modules"].get(name, name)
    resolved = (PACKAGE_ROOT / relative).resolve()
    if not resolved.is_relative_to(PACKAGE_ROOT):
        raise ValueError("Relocated module must remain inside the package.")
    return resolved


def current_code_digests():
    """Hash every package module, entry point and helper, plus the relocation map."""
    return {path.relative_to(PACKAGE_ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(PACKAGE_ROOT.rglob("*.py"))} | {
        "_relocation.json": sha256(RELOCATION_MANIFEST.read_bytes()).hexdigest()}


def relocated_digest_matches(name, recorded, current):
    """Recognize a baseline module moved without changing its calculation code."""
    if recorded == current:
        return True
    manifest = _manifest()
    if recorded is None or recorded != manifest["before"].get(name):
        return False
    relative = manifest["modules"].get(name, name)
    return current == manifest["after"].get(relative)
