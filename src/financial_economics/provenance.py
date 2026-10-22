"""Shared run-provenance helpers: file digests, git state and UTC timestamps."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import subprocess

from ._layout import current_code_digests


def file_digest(path):
    """SHA-256 of a file's bytes, as recorded in every run manifest."""
    return sha256(Path(path).read_bytes()).hexdigest()


def package_code_digests():
    """Actual SHA-256 of every package module and the relocation map."""
    return current_code_digests()


def directory_digests(directory):
    """SHA-256 of every file under a directory, keyed by its POSIX relative path."""
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): file_digest(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def verify_files(directory, files, message):
    """Raise ValueError(message) unless each recorded file is inside directory with its recorded SHA-256."""
    directory = Path(directory).resolve()
    for name, digest in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or file_digest(path) != digest:
            raise ValueError(message)


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def git_branch():
    return _git("branch", "--show-current")


def git_head():
    return _git("rev-parse", "HEAD")


def git_dirty():
    return bool(_git("status", "--porcelain"))


def git_state():
    """Commit and dirty flag, or ("unavailable", None) when git cannot be read."""
    try:
        return git_head(), git_dirty()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", None


def utc_now():
    return datetime.now(UTC).isoformat()
