#!/usr/bin/env python3
"""Audit the v26 PDF, source snapshot, and supplementary ZIP for identity leaks.

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
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fcswae" / "v26_evidence_manifest.json"
OUTPUT = ROOT / "fcswae" / "v26_anonymity_report.json"
SIDECAR = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
PDF = ROOT / "fcswae" / "sca_v26_iclr2027.pdf"
ARCHIVE = ROOT / "fcswae" / "sca_v26_iclr2027_supplementary.zip"
SELF = Path(__file__).resolve()

BASE_TARGETS = [
    "fcswae/sca_v26_iclr2027.tex",
    "fcswae/appendix_v26_iclr2027.tex",
    "fcswae/references.bib",
    "fcswae/README_V26_SUPPLEMENT.md",
    "fcswae/figures/FIGURE_SOURCES.md",
    "fcswae/build_sca_v26_submission.sh",
    "fcswae/v26_evidence_manifest.json",
    "fcswae/sca_v26_iclr2027_supplementary.zip",
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
        ROOT / "fcswae" / "sca_v26_iclr2027.tex",
        ROOT / "fcswae" / "appendix_v26_iclr2027.tex",
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


def scan_zip(path: Path, findings: list[dict[str, object]]) -> dict[str, object]:
    """Scan archive names/content and verify its internal content manifest."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            findings.append({"category": "duplicate_zip_entry", "path": relative(path)})

        entry_data: dict[str, bytes] = {}
        for name in names:
            location = f"{relative(path)}::{name}"
            if name.startswith("/") or ".." in Path(name).parts:
                findings.append({"category": "unsafe_zip_entry", "path": location})
            for category, pattern in GENERIC_PATTERNS.items():
                if pattern.search(name):
                    findings.append({"category": f"zip_name_{category}", "path": location})
            lowered_name = name.casefold()
            for category, token in dynamic_tokens().items():
                if token.casefold() in lowered_name:
                    findings.append({"category": f"zip_name_{category}", "path": location})

            data = archive.read(name)
            entry_data[name] = data
            suffix = Path(name).suffix.casefold()
            if suffix in {".png", ".jpg", ".jpeg", ".pdf", ".ckpt", ".pt", ".pth"}:
                lowered = data.lower()
                for category, token in {
                    "embedded_posix_user_directory": b"/home/",
                    "embedded_macos_user_directory": b"/users/",
                    "embedded_file_uri": b"file://",
                }.items():
                    if token in lowered:
                        findings.append({"category": f"zip_{category}", "path": location})
                continue

            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                findings.append({"category": "zip_unknown_binary", "path": location})
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                for category, pattern in GENERIC_PATTERNS.items():
                    if category == "email_address" and name.endswith(
                        ("iclr2027_conference.sty", "iclr2027_conference.bst")
                    ):
                        continue
                    if pattern.search(line):
                        findings.append(
                            {"category": f"zip_{category}", "path": location, "line": line_number}
                        )
                lowered = line.casefold()
                for category, token in dynamic_tokens().items():
                    if token.casefold() in lowered:
                        findings.append(
                            {"category": f"zip_{category}", "path": location, "line": line_number}
                        )

        manifest_names = [name for name in names if name.endswith("/CONTENT_MANIFEST.json")]
        if len(manifest_names) != 1:
            findings.append({"category": "zip_content_manifest_count", "path": relative(path)})
            return {"entries": len(names), "manifest_verified": False}

        content_manifest = json.loads(entry_data[manifest_names[0]])
        prefix = manifest_names[0].removesuffix("CONTENT_MANIFEST.json")
        verified = True
        for row in content_manifest.get("files", []):
            name = prefix + row["path"]
            data = entry_data.get(name)
            if data is None or len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
                verified = False
                findings.append({"category": "zip_content_hash_mismatch", "path": name})
        if len(content_manifest.get("files", [])) + 1 != len(names):
            verified = False
            findings.append({"category": "zip_unmanifested_entry", "path": relative(path)})
        return {"entries": len(names), "manifest_verified": verified}


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            "generate fcswae/v26_evidence_manifest.json before the anonymity audit"
        )
    target_names = sorted(
        set(BASE_TARGETS + manifest_targets() + manuscript_graphics())
    )
    targets = [ROOT / name for name in target_names]
    missing = [relative(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit targets: {missing}")

    findings: list[dict[str, object]] = []
    archive_summary: dict[str, object] = {"status": "not_scanned"}
    for path in targets:
        if path.resolve() == SELF:
            # This scanner necessarily contains literal examples of sensitive
            # path patterns. Its own source is pinned by the evidence manifest.
            continue
        if path.resolve() == ARCHIVE.resolve():
            archive_summary = scan_zip(path, findings)
        elif path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".pdf", ".ckpt", ".pt", ".pth"}:
            scan_binary(path, findings)
        else:
            scan_text(path, findings)

    metadata = pdf_metadata(findings)
    findings.sort(key=lambda item: (str(item["path"]), int(item.get("line", 0)), str(item["category"])))
    payload = {
        "schema": "iclr2027-v26-anonymity-audit-1.0.0",
        "status": "pass" if not findings else "fail",
        "audited_target_count": len(targets),
        "findings": findings,
        "pdf_metadata": metadata,
        "supplementary_archive": archive_summary,
        "checks": [
            "runtime username, home directory, and hostname",
            "POSIX/macOS/Windows user directories",
            "file URIs and email addresses",
            "submission PDF title, subject, keywords, and author metadata",
        ],
        "boundary": (
            "The audit covers v26 manuscript sources, included figures, the "
            "submission README/figure-source index, every source/artifact listed "
            "in the v26 evidence manifest, and every entry of the generated "
            "supplementary ZIP. Git object metadata is not packaged."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    SIDECAR.write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n")
    print(f"{payload['status']}: audited {len(targets)} targets; findings={len(findings)}")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
