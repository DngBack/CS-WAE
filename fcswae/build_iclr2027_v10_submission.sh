#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
main_tex="$script_dir/main_v10_iclr2027.tex"
build_log="$script_dir/main_v10_iclr2027.log"
pdf_file="$script_dir/main_v10_iclr2027.pdf"
style_file="$repo_dir/iclr2027/iclr2027_conference.sty"
bst_file="$repo_dir/iclr2027/iclr2027_conference.bst"

if [[ ! -f "$style_file" || ! -f "$bst_file" ]]; then
  echo "ERROR: official ICLR 2027 style files are missing from $repo_dir/iclr2027/." >&2
  echo "Do not build a submission PDF with the ICLR 2026 fallback." >&2
  exit 1
fi

if rg -q 'iclr2026' "$main_tex"; then
  echo "ERROR: manuscript source still references the ICLR 2026 template." >&2
  exit 1
fi

if ! rg -q '^\\section\*\{AI Use Statement\}' "$main_tex"; then
  echo "ERROR: required AI Use Statement is missing." >&2
  exit 1
fi

if ! rg -q '^\\section\*\{Reproducibility Statement\}' "$main_tex"; then
  echo "ERROR: Reproducibility Statement is missing." >&2
  exit 1
fi

if rg -q '^[[:space:]]*\\iclrfinalcopy' "$main_tex"; then
  echo "ERROR: \\iclrfinalcopy is enabled; the review submission must be anonymous." >&2
  exit 1
fi

(
  cd "$script_dir"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main_v10_iclr2027.tex
)

if ! rg -q '\.\./iclr2027/iclr2027_conference\.sty' "$build_log"; then
  echo "ERROR: build log does not confirm that the official ICLR 2027 style was loaded." >&2
  exit 1
fi

if rg -q 'undefined|multiply defined|Overfull|Official ICLR 2027 style unavailable' "$build_log"; then
  echo "ERROR: submission build log contains a blocking warning or error." >&2
  rg -n 'undefined|multiply defined|Overfull|Official ICLR 2027 style unavailable' "$build_log" >&2
  exit 1
fi

pdf_author=$(pdfinfo "$pdf_file" | sed -n 's/^Author:[[:space:]]*//p')
if [[ -n "$pdf_author" ]]; then
  echo "ERROR: PDF Author metadata is not blank: $pdf_author" >&2
  exit 1
fi

pdf_pages=$(pdfinfo "$pdf_file" | awk '/^Pages:/ {print $2}')
statement_page=0
for ((page = 1; page <= pdf_pages; page++)); do
  if pdftotext -f "$page" -l "$page" "$pdf_file" - | rg -q 'R[[:space:]]*EPRODUCIBILITY[[:space:]]+S[[:space:]]*TATEMENT'; then
    statement_page=$page
    break
  fi
done
if (( statement_page == 0 )); then
  echo "ERROR: could not locate the Reproducibility Statement in the PDF." >&2
  exit 1
fi
if (( statement_page > 10 )); then
  echo "ERROR: main text exceeds nine pages; statements begin on page $statement_page." >&2
  exit 1
fi

echo "Submission build passed automated template, statement, LaTeX, and PDF-metadata checks."
echo "Manually inspect pagination, visual layout, source anonymity, and the supplementary archive."
