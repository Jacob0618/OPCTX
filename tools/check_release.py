"""Offline release audit. Never prints matching content or reads credentials.

Only scans the supplied release tree and its Git blobs, if initialized. An
environment/credential file is rejected by path without opening its content.
No network, model calls, Git mutation, or original-workspace traversal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


PATTERNS = {
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{25,}"),
    "aq_api_key": re.compile(r"AQ\.[0-9A-Za-z_-]{25,}"),
    "openai_style_key": re.compile(r"\bsk-(?:proj-|svcacct-)?[0-9A-Za-z_-]{20,}"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,})"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    "credential_url": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
    "url_key_literal": re.compile(r"[?&](?:api[_-]?key|access_token|token)=[A-Za-z0-9_.-]{20,}", re.I),
    "bearer_literal": re.compile(r"\bBearer\s+[A-Za-z0-9_.-]{30,}", re.I),
}
ASSIGNMENT = re.compile(
    r"(?:[\"']?(?:api[_-]?key|api_token|access_token|refresh_token|client_secret|password|secret_key)[\"']?)"
    r"\s*[:=]\s*([\"'])([^\"'\r\n]{8,})\1", re.I)
PLACEHOLDERS = re.compile(r"^(?:YOUR[_ -]|EXAMPLE|PLACEHOLDER|DUMMY|TEST[_ -]|FAKE|<|\$\{|os\.environ|GEMINI_API_KEY$|OPENAI_API_KEY$)", re.I)
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_DIRS = {".venv", "venv", "node_modules", "runtime_deps", "cache", "caches", "logs"}
DATA_SUFFIXES = {".jsonl", ".csv", ".tsv", ".parquet", ".sqlite", ".db", ".npy", ".npz", ".pkl", ".pickle"}
BINARY_SUFFIXES = {".zip", ".7z", ".tar", ".gz", ".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg", ".exe", ".dll"}


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Git audit command failed (stderr suppressed)")
    return result.stdout


def credential_file(path):
    name = Path(path).name.lower()
    return ((name == ".env" or name.startswith(".env.")) and name not in {".env.example", ".env.template"}) or name in {
        "credentials.json", "credentials", "token.json", "secrets.json", "id_rsa", "id_ed25519"
    } or Path(path).suffix.lower() in {".pem", ".p12", ".pfx", ".key"}


def audit(root):
    root = root.resolve(strict=True)
    findings, files, total_bytes = [], 0, 0

    def report(scope, path, kind, line=None, severity="block"):
        item = dict(scope=scope, path=path, type=kind, severity=severity)
        if line is not None:
            item["line"] = line
        if item not in findings:
            findings.append(item)

    def scan(scope, name, content):
        try:
            text = content.decode("utf8")
        except UnicodeDecodeError:
            report(scope, name, "non_utf8_requires_review", severity="review")
            return
        for pattern_name, regex in PATTERNS.items():
            for match in regex.finditer(text):
                report(scope, name, pattern_name, text.count("\n", 0, match.start()) + 1)
        for match in ASSIGNMENT.finditer(text):
            if not PLACEHOLDERS.search(match[2]):
                report(scope, name, "possible_hardcoded_credential", text.count("\n", 0, match.start()) + 1,
                       severity="review")

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        name = rel.as_posix()
        if any(p in EXCLUDED_DIRS for p in rel.parts):
            continue
        if path.is_symlink():
            report("working_tree", name, "symlink_requires_review")
            continue
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            report("working_tree", name, "outside_release_path")
            continue
        files += 1
        size = path.stat().st_size
        total_bytes += size
        if credential_file(name):
            report("working_tree", name, "credential_file_not_read")
            continue
        if any(p.lower() in FORBIDDEN_DIRS for p in rel.parts):
            report("working_tree", name, "runtime_cache_or_dependencies")
        if path.suffix.lower() in DATA_SUFFIXES or path.suffix.lower() == ".log":
            report("working_tree", name, "dataset_log_or_binary_result", severity="review")
        if path.suffix.lower() in BINARY_SUFFIXES:
            report("working_tree", name, "binary_asset_requires_review", severity="review")
        if size > 10 * 1024 * 1024:
            report("working_tree", name, "large_file_not_read")
            continue
        scan("working_tree", name, path.read_bytes())

    tracked_n = history_blobs = commits_n = 0
    git_initialized = (root / ".git").exists()
    if git_initialized:
        actual_root = git(root, "rev-parse", "--show-toplevel").decode().strip()
        if Path(actual_root).resolve() != root:
            raise RuntimeError("Target is not the Git repository root")
        names = git(root, "ls-files", "-z").decode("utf8").split("\0")
        for name in filter(None, names):
            tracked_n += 1
            if credential_file(name):
                report("git_index", name, "credential_file_not_read")
                continue
            if any(p in EXCLUDED_DIRS | FORBIDDEN_DIRS for p in Path(name).parts):
                report("git_index", name, "forbidden_tracked_runtime_file")
            scan("git_index", name, git(root, "show", ":" + name))
        refs = git(root, "for-each-ref", "--format=%(refname)").decode().splitlines()
        if refs:
            commits_n = int(git(root, "rev-list", "--all", "--count").strip())
            for line in git(root, "rev-list", "--objects", "--all").decode("utf8").splitlines():
                pieces = line.split(" ", 1)
                oid, name = pieces[0], pieces[1] if len(pieces) == 2 else "(commit_or_tree)"
                if git(root, "cat-file", "-t", oid).strip() != b"blob":
                    continue
                history_blobs += 1
                if credential_file(name):
                    report("git_history", name, "credential_file_not_read")
                    continue
                size = int(git(root, "cat-file", "-s", oid).strip())
                if size > 10 * 1024 * 1024:
                    report("git_history", name, "large_blob_not_read")
                    continue
                scan("git_history", name, git(root, "cat-file", "blob", oid))
        # Inspect remote URL values without emitting them, never credential stores.
        result = subprocess.run(["git", "-C", str(root), "config", "--get-regexp", r"remote\..*\.url"],
                                capture_output=True, check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError("Remote configuration audit failed")
        scan("git_config", "remote_urls", result.stdout)

    return dict(root=str(root), files=files, total_bytes=total_bytes,
                git_initialized=git_initialized, tracked_files=tracked_n,
                commits=commits_n, history_blobs=history_blobs,
                severity_counts=dict(Counter(x["severity"] for x in findings)), findings=findings,
                note="No matched content is emitted; review findings require human inspection.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["severity_counts"].get("block", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
