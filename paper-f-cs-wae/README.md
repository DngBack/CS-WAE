# F-CS-WAE AAAI-27 Draft

This folder is a standalone manuscript workspace for the F-CS-WAE paper. It is intentionally separate from `paper/`, which contains the original CS-WAE manuscript.

## Files

- `main.tex` - full manuscript draft.
- `references.bib` - bibliography for the F-CS-WAE draft.
- `figures/` - copied figure assets used by the manuscript.
- `tables/` - reserved for exported tables if later generated from scripts.
- `submission_checklist.md` - experiment and writing checklist before AAAI-27 submission.

## Current Scientific Position

The draft treats CS-WAE as a simpler single-latent baseline and positions F-CS-WAE as the main factorized model.

Current evidence is strongest on CIFAR-10:

- F-CS-WAE, 3 seeds: ACC `80.87 +/- 0.52`, FID `83.04 +/- 0.86`.
- Available local baselines are much weaker under the current protocol.

MNIST exposes an important limitation:

- F-CS-WAE seed 0 improves clustering and reconstruction strongly.
- But naive independent prior sampling is worse than CS-WAE because the style latent still carries class information.
- Post-hoc representation-aware sampling fixes the class-consistency failure:
  - global Gaussian style sampling self-ACC: `0.15`
  - class-conditional diagonal style sampling with temperature `0.25` self-ACC: `1.00`

This is written as a core diagnostic insight plus a sampling fix. For an AAAI-level final submission, the next method step should be to integrate class-conditional style sampling into the evaluator and run full FID/multi-seed experiments.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or, if available:

```bash
latexmk -pdf main.tex
```

The current preamble uses a generic article style so it can compile without the official AAAI-27 author kit. Replace it with the official AAAI-27 style when released.

## Important Caveat

Do not submit this exact draft without running the missing experiments in `submission_checklist.md`. It now contains a concrete sampling fix and regenerated diagnostic figures, but still needs full FID and matched multi-seed ablations.
