# ICLR 2027 submission checklist

Policy source: https://iclr.cc/Conferences/2027/AuthorGuidelines

## Paper package

- [x] Genuine, informative abstract is present in the manuscript.
- [x] Anonymous author block; `\iclrfinalcopy` remains disabled.
- [x] Main text ends with the conclusion on page 9 in the current preview.
- [x] References precede the appendix.
- [x] Required AI Use Statement is present before references.
- [x] Reproducibility Statement is present before references.
- [x] Ethics Statement covers the public, non-sensitive benchmark setting.
- [x] Main claims and central figures appear within the nine-page main text.
- [x] Historical non-canonical probe tables remain excluded.
- [ ] Compile with the unmodified official ICLR 2027 style and recheck pages.
- [ ] Run `./build_iclr2027_submission.sh` successfully.
- [ ] Build from a clean tree and inspect the final PDF visually.
- [ ] Create and identity-audit the anonymous supplementary code archive.
- [ ] Remove author names, usernames, machine paths, repository remotes,
  checkpoint metadata, document metadata, acknowledgements, and identifying
  URLs from every supplementary file.

## OpenReview and deadlines

- [ ] Submit the genuine abstract by 2026-09-11 AOE.
- [ ] Freeze the author set at the abstract deadline; only reorder authors
  before the full-paper deadline if needed.
- [ ] Submit the paper and supplementary materials by 2026-09-16 23:59 AOE.
- [ ] Ensure every author has a current OpenReview profile and confirmed email.
- [ ] Identify at least one qualified reciprocal reviewer registered for at
  least three reviews, or verify that the new-group exemption applies.
- [ ] If any author appears on three or more submissions, verify the six-review
  requirement or an organizer exemption.
- [ ] Confirm compliance with the 20-paper author quota and the quota for papers
  without an eligible reviewer.
- [ ] Complete the OpenReview AI-use field consistently with the manuscript.
- [ ] Read and acknowledge the ICLR Code of Ethics in the submission workflow.

## Policy checks

- [ ] Confirm there is no concurrent substantially similar archival submission.
- [ ] Cite any related author preprints in the third person.
- [ ] Ensure anonymous links do not log or expose reviewer identity.
- [ ] Do not rely on appendix or rebuttal for evidence essential to the central
  claim.
- [ ] If revising during 2026-10-29--2026-11-11, keep changes scoped and explain
  them clearly because reviewers may ignore substantial additions.

## Current upstream template blocker

The official Author Guidelines link to
`https://github.com/ICLR/Master-Template/raw/master/iclr2027.zip`.  On
2026-08-11 this URL returned HTTP 404, and the official repository contained
templates only through ICLR 2026.  The manuscript therefore has a temporary
fallback solely for preview builds.  A fallback-generated PDF is not ready for
submission.
