"""Resolve relocated sources and recognize one exact, reviewed source relocation.

The manifest records real before/after byte hashes. New runs always record actual
current hashes. Historical whole-package fingerprints are accepted only while the
entire Python package matches the reviewed after-state; later edits invalidate that
exception. A historical single-module digest is accepted while that module itself
still matches its reviewed after-state, as selected-module provenance always did.
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


def _python_digests():
    return {path.relative_to(PACKAGE_ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(PACKAGE_ROOT.rglob("*.py"))}


def current_code_digests():
    """Hash nested implementations, entry points, helpers, and the relocation map."""
    return _python_digests() | {
        "_relocation.json": sha256(RELOCATION_MANIFEST.read_bytes()).hexdigest()}


def _reviewed_relocation():
    manifest = _manifest()
    return manifest if _python_digests() == manifest["after"] else None


def code_matches(recorded, current):
    """Compare exact fingerprints or the exact reviewed pre-relocation package."""
    if recorded == current:
        return True
    manifest = _reviewed_relocation()
    return bool(manifest and recorded == manifest["before"]
                and current == current_code_digests())


def relocated_digest_matches(name, recorded, current):
    """Recognize a baseline module moved without changing its calculation code.

    Only this module is compared, so edits elsewhere in the package do not affect it.
    """
    if recorded == current:
        return True
    manifest = _manifest()
    if recorded is None or recorded != manifest["before"].get(name):
        return False
    relative = manifest["modules"].get(name, name)
    return current == manifest["after"].get(relative)
