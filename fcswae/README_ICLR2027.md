# ICLR 2027 paper draft

The current anonymous submission draft consists of:

- main_v8_iclr2027.tex: main paper, bibliography, and appendix entrypoint.
- appendix_v8_iclr2027.tex: proofs, model details, canonical Stage-0
  protocol, full probe results, generation diagnostics, and evidence boundary.
- main_v8_iclr2027.pdf: current anonymous PDF built with the official ICLR
  2027 style and the 2026-08-11 external-swap/MMD/Gen-ACC update.

The main paper ends on page 9.  The page-limit-exempt Reproducibility, Ethics,
and AI Use statements occupy page 10, references begin on page 11, and the
appendix begins on page 12.

Build from the fcswae directory with:

    latexmk -pdf -interaction=nonstopmode -halt-on-error main_v8_iclr2027.tex

For the final submission build, use the stricter gate instead:

    ./build_iclr2027_submission.sh

It refuses to build unless the official 2027 style and bibliography files are
installed, and checks the required statements, anonymity mode, LaTeX log, and
PDF author metadata.

The source loads the unmodified official files at
../iclr2027/iclr2027_conference.{sty,bst} directly.  There is no fallback to an
older conference style; the strict build exits if either official file is
missing.

## Current evidence policy

The draft reports only measurements that can be tied to the canonical
Stage-0 artifacts or explicitly identified archived diagnostics:

- MNIST seed 0, Fashion-MNIST seed 0, and CIFAR-10 seeds 0/1/2.
- Stratified 60/20/20 held-out probes with train-only standardization.
- Separate posterior-mean and stochastic-posterior-sample views.
- Multiscale HSIC with 200 plus-one-calibrated permutations, including ten
  stochastic posterior draws for the term-4 conditional test.
- Global MMD with independent posterior/reference RNG streams and a
  500-permutation finite-sample null (`stage0-1.1.0`).
- External pixel-classifier Gen-ACC with 1,000 images per class and a
  training-only style bank.
- Independent pixel-classifier deterministic latent swaps with 500 donor
  draws per ordered class pair.
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
