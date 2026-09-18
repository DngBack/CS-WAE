# Anonymous ICLR 2027 v26 supplement

This archive accompanies the anonymous v26 submission. It contains the
submission PDF and LaTeX sources, the official ICLR 2027 style files needed to
rebuild the paper, critical audit/training source files, manuscript figures,
and the locally available result files selected by the content manifest.

The matched FACT evidence is packaged at seed level, including the direct
joint-test results, leakage audits, decoder interventions, aggregate summary,
and adverse outcomes. The complete-contract, kernel-robustness,
exact-marginal, SCFlow, cross-model, Shapes3D, entropy, and Stage-0 evidence is
included as the locally available JSON/CSV/figure subset described by
`CONTENT_MANIFEST.json`.

Large training checkpoints are not copied into this archive. Their SHA-256
identifiers remain in the result manifests so that the exact evaluated models
can be matched to the separately retained training archive. No claim in v26
depends on externally supplied five-pair or FACT-Lite aggregates; those files
are intentionally excluded because their seed-level artifacts are unavailable
in this snapshot.

Rebuild the packaged PDF from `fcswae/` with a standard TeX installation:

```text
latexmk -pdf -interaction=nonstopmode -halt-on-error sca_v26_iclr2027.tex
```

In the full project checkout, `./fcswae/build_sca_v26_submission.sh`
regenerates the data-driven FACT figure, evidence manifest, PDF, and
supplementary ZIP, then audits both the filesystem inputs and every ZIP entry
for common identity leaks. The generated anonymity report and SHA-256 sidecars
sit next to the submission files outside this archive. The Python environment
and large checkpoints are deliberately not vendored into this compact ZIP.
