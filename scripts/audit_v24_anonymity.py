#!/usr/bin/env python3
"""Audit the v24 submission-facing snapshot for common identity leaks.

The report records only finding categories and repository-relative locations;
it never copies a potentially identifying token into the report itself.
"""

from __future__ import annotations

import getpass
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fcswae" / "v24_evidence_manifest.json"
OUTPUT = ROOT / "fcswae" / "v24_anonymity_report.json"
SIDECAR = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
PDF = ROOT / "fcswae" / "sca_v24_iclr2027.pdf"
SELF = Path(__file__).resolve()

BASE_TARGETS = [
    "fcswae/sca_v24_iclr2027.tex",
    "fcswae/appendix_v24_iclr2027.tex",
    "fcswae/references.bib",
    "fcswae/README_ICLR2027.md",
    "fcswae/figures/FIGURE_SOURCES.md",
    "fcswae/build_sca_v24_submission.sh",
    "fcswae/v24_evidence_manifest.json",
]

GENERIC_PATTERNS = {
    "posix_user_directory": re.compile(
        r"(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s\"'<>]+"
    ),
    "windows_user_directory": re.compile(
        r"[A-Za-z]:\\Users\\[^\\\s\"'<>]+", re.IGNORECASE
    ),
    "file_uri": re.compile(r"file://", re.IGNORECASE),
    "email_address": re.compile(
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9.-])"
    ),
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def manifest_targets() -> list[str]:
    payload = json.loads(MANIFEST.read_text())
    paths: list[str] = []
    for group in ("critical_source_snapshot", "headline_artifacts"):
        paths.extend(item["path"] for item in payload[group])
    return paths


def manuscript_graphics() -> list[str]:
    graphics: list[str] = []
    pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    for manuscript in (
        ROOT / "fcswae" / "sca_v24_iclr2027.tex",
        ROOT / "fcswae" / "appendix_v24_iclr2027.tex",
    ):
        for name in pattern.findall(manuscript.read_text()):
            candidate = ROOT / "fcswae" / "figures" / name
            if not candidate.suffix:
                for extension in (".pdf", ".png", ".jpg", ".jpeg"):
                    extended = candidate.with_suffix(extension)
                    if extended.is_file():
                        candidate = extended
                        break
            graphics.append(relative(candidate))
    return graphics


def dynamic_tokens() -> dict[str, str]:
    candidates = {
        "runtime_home": str(Path.home()),
        "runtime_username": getpass.getuser(),
        "runtime_hostname": socket.gethostname(),
    }
    return {
        category: token
        for category, token in candidates.items()
        if len(token) >= 4 and token not in {"root", "localhost", "unknown"}
    }


def scan_text(path: Path, findings: list[dict[str, object]]) -> None:
    try:
        text = path.read_text(errors="replace")
    except OSError as error:
        findings.append(
            {"category": "unreadable_target", "path": relative(path), "detail": type(error).__name__}
        )
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        for category, pattern in GENERIC_PATTERNS.items():
            if pattern.search(line):
                findings.append(
                    {"category": category, "path": relative(path), "line": line_number}
                )
        lowered = line.casefold()
        for category, token in dynamic_tokens().items():
            if token.casefold() in lowered:
                findings.append(
                    {"category": category, "path": relative(path), "line": line_number}
                )


def scan_binary(path: Path, findings: list[dict[str, object]]) -> None:
    try:
        data = path.read_bytes()
    except OSError as error:
        findings.append(
            {"category": "unreadable_target", "path": relative(path), "detail": type(error).__name__}
        )
        return
    lowered = data.lower()
    byte_tokens = {
        "embedded_posix_user_directory": b"/home/",
        "embedded_macos_user_directory": b"/users/",
        "embedded_file_uri": b"file://",
    }
    for category, token in byte_tokens.items():
        if token in lowered:
            findings.append({"category": category, "path": relative(path)})
    for category, token in dynamic_tokens().items():
        encoded = token.casefold().encode(errors="ignore")
        if encoded and encoded in lowered:
            findings.append({"category": category, "path": relative(path)})


def pdf_metadata(findings: list[dict[str, object]]) -> dict[str, str]:
    if not PDF.is_file():
        return {"status": "not_built"}
    completed = subprocess.run(
        ["pdfinfo", str(PDF)], check=True, capture_output=True, text=True
    )
    selected: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"Title", "Subject", "Keywords", "Author", "Creator", "Producer"}:
            selected[key.casefold()] = value.strip()
    for key in ("title", "subject", "keywords", "author"):
        if selected.get(key):
            findings.append({"category": f"pdf_{key}_metadata", "path": relative(PDF)})
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            "generate fcswae/v24_evidence_manifest.json before the anonymity audit"
        )
    target_names = sorted(
        set(BASE_TARGETS + manifest_targets() + manuscript_graphics())
    )
    targets = [ROOT / name for name in target_names]
    missing = [relative(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit targets: {missing}")

    findings: list[dict[str, object]] = []
    for path in targets:
        if path.resolve() == SELF:
            # This scanner necessarily contains literal examples of sensitive
            # path patterns. Its own source is pinned by the evidence manifest.
            continue
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".pdf", ".ckpt", ".pt", ".pth"}:
            scan_binary(path, findings)
        else:
            scan_text(path, findings)

    metadata = pdf_metadata(findings)
    findings.sort(key=lambda item: (str(item["path"]), int(item.get("line", 0)), str(item["category"])))
    payload = {
        "schema": "iclr2027-v24-anonymity-audit-1.0.0",
        "status": "pass" if not findings else "fail",
        "audited_target_count": len(targets),
        "findings": findings,
        "pdf_metadata": metadata,
        "checks": [
            "runtime username, home directory, and hostname",
            "POSIX/macOS/Windows user directories",
            "file URIs and email addresses",
            "submission PDF title, subject, keywords, and author metadata",
        ],
        "boundary": (
            "The audit covers v24 manuscript sources, included figures, the "
            "submission README/figure-source index, and every source/artifact "
            "listed in the v24 evidence manifest. It does not certify files that "
            "are later added to a supplementary archive. Git object metadata "
            "must not be included in the anonymous archive."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    SIDECAR.write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(f"{payload['status']}: audited {len(targets)} targets; findings={len(findings)}")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
