#!/usr/bin/env python3
"""Verify the final ICLR submission folder and write its audit/checksums."""

from __future__ import annotations

from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper.pdf"
SOURCE_ZIP = ROOT / "sca_v26_latex_source.zip"
EVIDENCE_ZIP = ROOT / "investigation/sca_v26_iclr2027_supplementary.zip"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_matches(path: Path, sidecar: Path) -> bool:
    declared = sidecar.read_text().split()[0]
    return declared == sha256_file(path)


def pdf_summary() -> dict[str, object]:
    raw = subprocess.run(
        ["pdfinfo", str(PAPER)], check=True, capture_output=True, text=True
    ).stdout
    metadata = {}
    for line in raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.casefold()] = value.strip()
    text = subprocess.run(
        ["pdftotext", str(PAPER), "-"], check=True, capture_output=True, text=True
    ).stdout
    pages = int(metadata["pages"])
    statement_page = 0
    for page in range(1, pages + 1):
        page_text = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), str(PAPER), "-"],
            check=True, capture_output=True, text=True,
        ).stdout
        if re.search(r"R\s*EPRODUCIBILITY\s+S\s*TATEMENT", page_text):
            statement_page = page
            break
    return {
        "pages": pages,
        "reproducibility_statement_page": statement_page,
        "author_metadata_blank": not metadata.get("author"),
        "title_metadata_blank": not metadata.get("title"),
        "ai_use_statement_present": bool(
            re.search(r"AI\s+U\s*SE\s+S\s*TATEMENT", text.upper())
        ),
    }


def safe_zip(archive: zipfile.ZipFile) -> bool:
    names = archive.namelist()
    return len(names) == len(set(names)) and all(
        not name.startswith("/") and ".." not in Path(name).parts for name in names
    )


def scan_archive(archive: zipfile.ZipFile) -> list[dict[str, str]]:
    findings = []
    runtime_tokens = [getpass.getuser(), socket.gethostname(), str(Path.home())]
    for name in archive.namelist():
        data = archive.read(name)
        lowered = data.lower()
        for token in runtime_tokens:
            if len(token) >= 4 and token.casefold().encode() in lowered:
                findings.append({"category": "runtime_identity", "entry": name})
        if Path(name).suffix.casefold() in {".png", ".jpg", ".jpeg", ".pdf"}:
            if b"/home/" in lowered or b"file://" in lowered:
                findings.append({"category": "binary_local_path", "entry": name})
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        # Scanner/build sources intentionally contain generic leak patterns;
        # exact runtime identity tokens above are still checked in every entry.
        if not name.endswith(("build.sh", "verify_package.py")):
            if re.search(r"/(?:home|Users)/[^/\s\"']+", text) or "file://" in text.casefold():
                findings.append({"category": "text_local_path", "entry": name})
    return findings


def source_summary() -> dict[str, object]:
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        names = archive.namelist()
        main = archive.read("main.tex").decode()
        appendix = archive.read("appendix.tex").decode()
        graphics = re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main + appendix
        )
        missing_graphics = [
            name for name in graphics if f"figures/{name}" not in names
        ]
        finalcopy_enabled = bool(re.search(r"^[ \t]*\\iclrfinalcopy", main, re.MULTILINE))
        rejected_face_direction_absent = not re.search(
            r"UTKFace|CelebA-HQ|FFHQ", main + appendix, re.IGNORECASE
        )
        return {
            "entries": len(names),
            "safe_paths_and_unique_names": safe_zip(archive),
            "missing_graphics": missing_graphics,
            "anonymous_author_source": r"\author{Anonymous Authors}" in main,
            "finalcopy_disabled": not finalcopy_enabled,
            "official_2027_style_present": "iclr2027_conference.sty" in names,
            "rejected_face_direction_absent": rejected_face_direction_absent,
            "identity_findings": scan_archive(archive),
        }


def evidence_summary() -> dict[str, object]:
    with zipfile.ZipFile(EVIDENCE_ZIP) as archive:
        names = archive.namelist()
        manifest_names = [name for name in names if name.endswith("/CONTENT_MANIFEST.json")]
        verified = len(manifest_names) == 1
        excluded = []
        if verified:
            manifest_name = manifest_names[0]
            manifest = json.loads(archive.read(manifest_name))
            prefix = manifest_name.removesuffix("CONTENT_MANIFEST.json")
            excluded = manifest.get("excluded_untraceable_aggregates", [])
            for row in manifest.get("files", []):
                name = prefix + row["path"]
                if name not in names:
                    verified = False
                    break
                data = archive.read(name)
                if len(data) != row["bytes"] or sha256_bytes(data) != row["sha256"]:
                    verified = False
                    break
            if len(manifest.get("files", [])) + 1 != len(names):
                verified = False
        decoded_text = ""
        for name in names:
            if Path(name).suffix.casefold() in {".png", ".jpg", ".jpeg", ".pdf"}:
                continue
            try:
                decoded_text += archive.read(name).decode("utf-8")
            except UnicodeDecodeError:
                pass
        return {
            "entries": len(names),
            "safe_paths_and_unique_names": safe_zip(archive),
            "internal_manifest_verified": verified,
            "identity_findings": scan_archive(archive),
            "rejected_face_direction_absent": not re.search(
                r"UTKFace|CelebA-HQ|FFHQ", decoded_text, re.IGNORECASE
            ),
            "excluded_untraceable_aggregates": excluded,
        }


def main() -> None:
    required = [
        PAPER, SOURCE_ZIP, EVIDENCE_ZIP,
        ROOT / "investigation/sca_v26_iclr2027_supplementary.zip.sha256",
        ROOT / "investigation/v26_anonymity_report.json",
        ROOT / "investigation/v26_anonymity_report.json.sha256",
        ROOT / "investigation/v26_evidence_manifest.json",
        ROOT / "investigation/v26_evidence_manifest.json.sha256",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    pdf = pdf_summary()
    source = source_summary()
    evidence = evidence_summary()
    copied_audit = json.loads((ROOT / "investigation/v26_anonymity_report.json").read_text())
    sidecars = {
        "evidence_zip": sidecar_matches(
            EVIDENCE_ZIP,
            ROOT / "investigation/sca_v26_iclr2027_supplementary.zip.sha256",
        ),
        "anonymity_report": sidecar_matches(
            ROOT / "investigation/v26_anonymity_report.json",
            ROOT / "investigation/v26_anonymity_report.json.sha256",
        ),
        "evidence_manifest": sidecar_matches(
            ROOT / "investigation/v26_evidence_manifest.json",
            ROOT / "investigation/v26_evidence_manifest.json.sha256",
        ),
    }
    checks = [
        pdf["pages"] == 30,
        pdf["reproducibility_statement_page"] == 10,
        pdf["author_metadata_blank"], pdf["title_metadata_blank"],
        pdf["ai_use_statement_present"],
        source["safe_paths_and_unique_names"], not source["missing_graphics"],
        source["anonymous_author_source"], source["finalcopy_disabled"],
        source["official_2027_style_present"], source["rejected_face_direction_absent"],
        not source["identity_findings"],
        evidence["safe_paths_and_unique_names"], evidence["internal_manifest_verified"],
        evidence["rejected_face_direction_absent"], not evidence["identity_findings"],
        copied_audit.get("status") == "pass", not copied_audit.get("findings"),
        all(sidecars.values()),
    ]
    report = {
        "schema": "iclr2027-submission-folder-audit-1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks) else "fail",
        "pdf": pdf,
        "source_archive": source,
        "investigation_archive": evidence,
        "copied_v26_anonymity_audit_status": copied_audit.get("status"),
        "sidecar_hashes_verified": sidecars,
    }
    (ROOT / "PACKAGE_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    checksum_targets = [
        PAPER, SOURCE_ZIP, EVIDENCE_ZIP,
        ROOT / "investigation/v26_anonymity_report.json",
        ROOT / "investigation/v26_evidence_manifest.json",
        ROOT / "PACKAGE_AUDIT.json",
    ]
    lines = [
        f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in checksum_targets
    ]
    (ROOT / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
