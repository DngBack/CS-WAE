#!/usr/bin/env python3
"""Build a deterministic, identity-sanitized v26 supplementary archive."""

from __future__ import annotations

import hashlib
import getpass
import json
from pathlib import Path
import re
import socket
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "fcswae" / "sca_v26_iclr2027_supplementary.zip"
SIDECAR = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
PREFIX = "sca_v26_supplementary"

TEXT_SUFFIXES = {
    ".bib", ".csv", ".json", ".md", ".py", ".sha256", ".sh",
    ".sty", ".bst", ".tex", ".txt", ".yaml", ".yml",
}

EXPLICIT_FILES = [
    "fcswae/sca_v26_iclr2027.pdf",
    "fcswae/sca_v26_iclr2027.tex",
    "fcswae/appendix_v26_iclr2027.tex",
    "fcswae/references.bib",
    "fcswae/build_sca_v26_submission.sh",
    "fcswae/README_V26_SUPPLEMENT.md",
    "fcswae/v26_evidence_manifest.json",
    "fcswae/v26_evidence_manifest.json.sha256",
    "fcswae/figures/FIGURE_SOURCES.md",
    "iclr2027/iclr2027_conference.sty",
    "iclr2027/iclr2027_conference.bst",
    "scripts/build_v26_evidence_manifest.py",
    "scripts/build_v26_supplement.py",
    "scripts/plot_v26_fact_tradeoff.py",
    "scripts/audit_mnist_joint_contract.py",
    "scripts/summarize_kernel_robustness.py",
    "scripts/scflow_feasibility_gate.py",
    "scripts/run_scflow_contract_audit.py",
    "scripts/summarize_scflow_audit.py",
    "src/metrics/audit_protocol.py",
    "src/metrics/deadiff_adapter.py",
    "src/trainers/trainer_f_cs_wae.py",
    "src/utils/loss_f_cs_wae.py",
    "tests/test_audit_protocol.py",
    "tests/test_deadiff_adapter.py",
    "fcswae/reported_followup_aggregates_v16.json",
]

# Trees are restricted to compact, review-relevant file types. In particular,
# checkpoints and datasets are not silently pulled into the submission ZIP.
TREE_RULES = [
    ("runs_diag/fact/mean_hsic_mnist_40e/submission_followup_v1", None),
    ("runs_diag/complete_contract_diagnostics", None),
    ("runs_diag/kernel_robustness", {".json", ".csv", ".sha256"}),
    ("runs_diag/synthetic_exact_marginal_paper", None),
    ("runs_modern/scflow_audit", {".json", ".csv", ".sha256"}),
    ("runs_modern/scflow_feasibility", {".json", ".csv", ".sha256"}),
    ("runs_cross_model/native_v1", {".json", ".csv", ".sha256"}),
    ("runs_shapes3d/factor_audit_v1", {".json", ".csv", ".sha256"}),
    ("runs_f/entropy_sweep", {".json", ".csv", ".sha256"}),
    ("runs_diag/stage0", {".json", ".csv", ".sha256"}),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manuscript_graphics() -> list[str]:
    pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    selected: list[str] = []
    for manuscript in (
        ROOT / "fcswae/sca_v26_iclr2027.tex",
        ROOT / "fcswae/appendix_v26_iclr2027.tex",
    ):
        for name in pattern.findall(manuscript.read_text(encoding="utf-8")):
            candidate = ROOT / "fcswae/figures" / name
            if not candidate.suffix:
                for suffix in (".pdf", ".png", ".jpg", ".jpeg"):
                    extended = candidate.with_suffix(suffix)
                    if extended.is_file():
                        candidate = extended
                        break
            selected.append(candidate.relative_to(ROOT).as_posix())
    return selected


def selected_paths() -> list[Path]:
    relative_names = set(EXPLICIT_FILES + manuscript_graphics())
    for relative_root, suffixes in TREE_RULES:
        root = ROOT / relative_root
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes):
                relative_names.add(path.relative_to(ROOT).as_posix())
    paths = [ROOT / name for name in sorted(relative_names)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing supplement inputs: {missing}")
    return paths


def sanitized_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return data
    text = data.decode("utf-8", errors="strict")
    root_prefix = str(ROOT.resolve()) + "/"
    text = text.replace(root_prefix, "repository/")
    # Historical JSON may embed the same checkout under a different user root.
    text = re.sub(r"/(?:home|Users)/[^/\s\"']+/", "user-root/", text)
    text = text.replace(getpass.getuser(), "anonymous-user")
    text = text.replace(socket.gethostname(), "anonymous-host")
    return text.encode("utf-8")


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 2, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def main() -> None:
    entries: list[tuple[str, bytes]] = []
    manifest_rows: list[dict[str, object]] = []
    for path in selected_paths():
        relative = path.relative_to(ROOT).as_posix()
        data = sanitized_bytes(path)
        archive_name = f"{PREFIX}/{relative}"
        entries.append((archive_name, data))
        manifest_rows.append({"path": relative, "bytes": len(data), "sha256": sha256(data)})

    manifest = {
        "schema": "iclr2027-v26-supplement-content-1.0.0",
        "checkpoint_files_included": False,
        "excluded_untraceable_aggregates": ["external five-pair", "FACT-Lite"],
        "files": manifest_rows,
    }
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    entries.append((f"{PREFIX}/CONTENT_MANIFEST.json", manifest_data))

    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name, data in sorted(entries):
            archive.writestr(zip_info(name), data)
    temporary.replace(OUTPUT)
    SIDECAR.write_text(f"{sha256(OUTPUT.read_bytes())}  {OUTPUT.name}\n", encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(entries)} entries, {OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
