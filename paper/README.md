# CS-WAE Paper Draft

This folder contains a simple arXiv-style LaTeX draft for the current CS-WAE results.

## Files

- `main.tex` - main paper plus appendix
- `references.bib` - bibliography
- `figures/` - local PNG copies of all figures used by the paper

The paper is now self-contained with respect to figures, so compile from this folder. All included figures use PNG files.

## Build

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The draft uses common LaTeX packages: `natbib`, `booktabs`, `subcaption`, `algorithm`, and `algpseudocode`/`algorithmicx`.
If your TeX installation is minimal, install the missing packages through MiKTeX Console or TeX Live Manager.

## Notes

- The current framing is **label-guided generative clustering**, not fully unsupervised clustering, because training uses labels in the supervised MMD term.
- Baseline comparisons are currently seed 0 only; this is stated as a limitation.
- MNIST ablation uses an older training path; this is also stated explicitly.
- The main text has been expanded with design rationale, training algorithm, per-seed analysis, reconstruction/generation figures, ablation reconstructions, and sample-quality discussion.
