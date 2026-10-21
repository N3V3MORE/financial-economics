"""Shared run-provenance helpers: file digests, git state and UTC timestamps."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
import subprocess


def file_digest(path):
    """SHA-256 of a file's bytes, as recorded in every run manifest."""
    return sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def git_branch():
    return _git("branch", "--show-current")


def git_head():
    return _git("rev-parse", "HEAD")


def git_dirty():
    return bool(_git("status", "--porcelain"))


def utc_now():
    return datetime.now(UTC).isoformat()
