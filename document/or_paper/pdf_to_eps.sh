#!/bin/bash
# Usage: ./pdf_to_eps.sh input.pdf [output_prefix]
# Converts each page of input.pdf to output_prefix_p1.eps, _p2.eps, ...
# Requires: ghostscript (gs), pdfcrop (TeX Live)

set -e

INPUT="${1:?Usage: $0 input.pdf [output_prefix]}"
PREFIX="${2:-$(basename "$INPUT" .pdf)}"
TMPFILE="/tmp/_pdfcrop_$$.pdf"

pdfcrop --margins 5 "$INPUT" "$TMPFILE"

N_PAGES=$(python3 -c "
import subprocess, re
r = subprocess.run(['mdls', '-name', 'kMDItemNumberOfPages', '$INPUT'], capture_output=True, text=True)
print(r.stdout.split('=')[1].strip())
")

for i in $(seq 1 "$N_PAGES"); do
    OUT="${PREFIX}_p${i}.eps"
    gs -dNOPAUSE -dBATCH -dQUIET -sDEVICE=eps2write \
       -dFirstPage="$i" -dLastPage="$i" \
       -sOutputFile="$OUT" "$TMPFILE"
    echo "Saved: $OUT"
done

rm -f "$TMPFILE"
