# ICLR 2027 submission package — SCA v26

This directory is a standalone, anonymous submission snapshot.  It deliberately
excludes the exploratory UTKFace experiment because that direction was rejected
after the paper evidence boundary was frozen.

## Upload artifacts

- Upload `paper.pdf` as the paper; it already includes the investigation appendix after the references.
- Upload `investigation/sca_v26_iclr2027_supplementary.zip` only if using the supplementary-code/material field.
- Keep `sca_v26_latex_source.zip` as the self-contained source snapshot, or upload it only if the portal explicitly requests LaTeX source.

## Rebuild

```bash
cd source
./build.sh
```

The build gate checks anonymity mode, local-path leaks, LaTeX warnings, blank PDF
author metadata, and the nine-page main-text boundary.  `PACKAGE_AUDIT.json`
records checks performed on the final folder and both ZIP archives.

Checked against the official ICLR 2027 Author Guidelines on 2026-09-03: review
submissions are double blind; main text is limited to nine pages; appendices may
follow the references in the same PDF; and an AI use statement is required.
The abstract deadline is 2026-09-18 11:59 PM AOE and the paper/supplement deadline
is 2026-09-25 11:59 PM AOE.  Confirm the live OpenReview upload fields and file-size
limits immediately before upload.

Official guide: https://iclr.cc/Conferences/2027/AuthorGuidelines
