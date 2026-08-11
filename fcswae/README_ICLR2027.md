# ICLR 2027 paper draft

The current anonymous submission draft consists of:

- main_v8_iclr2027.tex: main paper, bibliography, and appendix entrypoint.
- appendix_v8_iclr2027.tex: proofs, model details, canonical Stage-0
  protocol, full probe results, generation diagnostics, and evidence boundary.
- main_v8_iclr2027.pdf: compiled anonymous preview.

The compiled layout uses nine pages for the main paper.  The page-limit-exempt
Reproducibility, Ethics, and AI Use statements follow on a separate page,
before the references.  The ninth main-text page is used for evidence-backed
diagnostic figures and discussion rather than non-canonical historical probes.

Build from the fcswae directory with:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main_v8_iclr2027.tex

For the final submission build, use the stricter gate instead:

    ./build_iclr2027_submission.sh

It refuses to build unless the official 2027 style and bibliography files are
installed, and checks the required statements, anonymity mode, LaTeX log, and
PDF author metadata.

The source first looks for the unmodified official files at
../iclr2027/iclr2027_conference.{sty,bst}.  If they are absent, it falls back
to the repository's ICLR 2026 shell and emits a build warning.  As of
2026-08-11, the official URL published by the ICLR 2027 Author Guidelines
returns HTTP 404 and ICLR/Master-Template does not contain 2027 files.  Do not
submit a PDF produced by the fallback: install the official archive at
../iclr2027/ and recompile once upstream is corrected.

## Current evidence policy

The draft reports only measurements that can be tied to the canonical
Stage-0 artifacts or explicitly identified archived diagnostics:

- MNIST seed 0, Fashion-MNIST seed 0, and CIFAR-10 seeds 0/1/2.
- Stratified 60/20/20 held-out probes with train-only standardization.
- Separate posterior-mean and stochastic-posterior-sample views.
- Multiscale HSIC with 200 plus-one-calibrated permutations.
- External pixel-classifier Gen-ACC from the Stage-0 JSON artifacts.
- Per-class and pairwise conditional-MMD figures generated from stochastic
  style samples, plus qualitative sampling/swap grids labeled with their
  evaluator limitations.

Earlier cross-model probe, multi-remedy, five-seed, and explicit split-latent
tables remain excluded until they are regenerated with the canonical protocol
and complete manifests.

## Submission checks

- Keep \iclrfinalcopy commented for review.
- Keep the author block anonymous in both source and rendered PDF.
- Retain the required AI Use Statement and complete the corresponding
  OpenReview disclosure field consistently.
- Upload code only through an identity-audited anonymous ZIP/repository; do
  not add an identifying URL to the manuscript.
- Compile from a clean checkout before submission and inspect the log for
  undefined citations/references or overfull boxes.
- Confirm that the build log no longer contains the ICLR 2026 fallback warning
  before submission.
