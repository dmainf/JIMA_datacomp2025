#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT="$DIR/figure.pdf"
TMPFILE="/tmp/_pdfcrop_$$.pdf"

pdfcrop --margins 5 "$INPUT" "$TMPFILE"

gs -dNOPAUSE -dBATCH -dQUIET -sDEVICE=eps2write \
   -dFirstPage=1 -dLastPage=1 \
   -sOutputFile="$DIR/architecture_ProtoRAF.eps" "$TMPFILE"
echo "Saved: architecture_ProtoRAF.eps"
gs -dNOPAUSE -dBATCH -dQUIET -sDEVICE=pdfwrite \
   -dFirstPage=1 -dLastPage=1 \
   -sOutputFile="$DIR/architecture_ProtoRAF.pdf" "$TMPFILE"
echo "Saved: architecture_ProtoRAF.pdf"

gs -dNOPAUSE -dBATCH -dQUIET -sDEVICE=eps2write \
   -dFirstPage=2 -dLastPage=2 \
   -sOutputFile="$DIR/architecture_GateRAF.eps" "$TMPFILE"
echo "Saved: architecture_GateRAF.eps"
gs -dNOPAUSE -dBATCH -dQUIET -sDEVICE=pdfwrite \
   -dFirstPage=2 -dLastPage=2 \
   -sOutputFile="$DIR/architecture_GateRAF.pdf" "$TMPFILE"
echo "Saved: architecture_GateRAF.pdf"

rm -f "$TMPFILE"
