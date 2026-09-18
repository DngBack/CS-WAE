#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$source_dir"

for required in main.tex appendix.tex references.bib \
  iclr2027_conference.sty iclr2027_conference.bst; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing $required" >&2
    exit 1
  fi
done

if rg -q '^[[:space:]]*\\iclrfinalcopy' main.tex; then
  echo 'ERROR: review source enables \iclrfinalcopy.' >&2
  exit 1
fi
if ! rg -q '^\\author\{Anonymous Authors\}' main.tex; then
  echo 'ERROR: source is not explicitly anonymous.' >&2
  exit 1
fi
if rg -q '/home/|/Users/|file://' --glob '*.tex' --glob '*.bib' .; then
  echo 'ERROR: local path leaked into text source.' >&2
  exit 1
fi

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

if rg -q 'undefined|multiply defined|Overfull|LaTeX Error' main.log; then
  echo 'ERROR: blocking LaTeX warning/error found.' >&2
  rg -n 'undefined|multiply defined|Overfull|LaTeX Error' main.log >&2
  exit 1
fi

pdf_author=$(pdfinfo main.pdf | sed -n 's/^Author:[[:space:]]*//p')
if [[ -n "$pdf_author" ]]; then
  echo "ERROR: PDF Author metadata is not blank: $pdf_author" >&2
  exit 1
fi

pages=$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')
statement_page=0
for ((page=1; page<=pages; page++)); do
  if pdftotext -f "$page" -l "$page" main.pdf - | \
      rg -q 'R[[:space:]]*EPRODUCIBILITY[[:space:]]+S[[:space:]]*TATEMENT'; then
    statement_page=$page
    break
  fi
done
if (( statement_page == 0 || statement_page > 10 )); then
  echo "ERROR: reproducibility statement starts on invalid page $statement_page" >&2
  exit 1
fi

echo "PASS: standalone anonymous ICLR source built ($pages pages; statements begin page $statement_page)."
