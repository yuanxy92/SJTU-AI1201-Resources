#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/export_md_pdf.sh [input.md] [output.pdf]
  scripts/export_md_pdf.sh [chapters_dir] [output.pdf]

Examples:
  scripts/export_md_pdf.sh
  scripts/export_md_pdf.sh output/期末复习课详细提纲.md
  scripts/export_md_pdf.sh output/chapters
  scripts/export_md_pdf.sh output/期末复习课详细提纲.md output/期末复习课详细提纲.pdf

Notes:
  Requires pandoc and xelatex.
  Install pandoc with: brew install pandoc
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_INPUT="$ROOT_DIR/output/chapters"
DEFAULT_OUTPUT="$ROOT_DIR/output/期末复习课详细提纲.pdf"

INPUT="${1:-$DEFAULT_INPUT}"
if [[ "$INPUT" != /* ]]; then
  INPUT="$ROOT_DIR/$INPUT"
fi

if [[ ! -f "$INPUT" && ! -d "$INPUT" ]]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

if [[ -n "${2:-}" ]]; then
  OUTPUT="$2"
  if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$ROOT_DIR/$OUTPUT"
  fi
else
  if [[ -d "$INPUT" ]]; then
    OUTPUT="$DEFAULT_OUTPUT"
  else
    OUTPUT="${INPUT%.md}.pdf"
  fi
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is not installed." >&2
  echo "Install it with: brew install pandoc" >&2
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "xelatex is not installed." >&2
  echo "Install a TeX distribution, for example MacTeX." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$ROOT_DIR/tmp"

PANDOC_INPUT="$INPUT"
TEMP_INPUT=""
if [[ -d "$INPUT" ]]; then
  CHAPTERS=()
  while IFS= read -r chapter; do
    CHAPTERS+=("$chapter")
  done < <(find "$INPUT" -maxdepth 1 -type f -name '*.md' | sort)
  if [[ "${#CHAPTERS[@]}" -eq 0 ]]; then
    echo "No markdown chapters found in: $INPUT" >&2
    exit 1
  fi

  TEMP_INPUT="$(mktemp "$ROOT_DIR/tmp/combined-review.XXXXXX.md")"
  for chapter in "${CHAPTERS[@]}"; do
    cat "$chapter" >> "$TEMP_INPUT"
    printf '\n\n' >> "$TEMP_INPUT"
  done
  PANDOC_INPUT="$TEMP_INPUT"
fi

cleanup() {
  if [[ -n "$TEMP_INPUT" && -f "$TEMP_INPUT" ]]; then
    rm -f "$TEMP_INPUT"
  fi
}
trap cleanup EXIT

FONT="${CJK_FONT:-Songti SC}"
MONO_FONT="${MONO_FONT:-Menlo}"
MARGIN="${PDF_MARGIN:-0.85in}"

echo "Exporting:"
echo "  Input : $INPUT"
if [[ -d "$INPUT" ]]; then
  echo "  Mode  : combine chapters"
fi
echo "  Output: $OUTPUT"
echo "  Font  : $FONT"

pandoc "$PANDOC_INPUT" \
  --from markdown+smart \
  --to pdf \
  --output "$OUTPUT" \
  --pdf-engine=xelatex \
  --resource-path="$ROOT_DIR:$ROOT_DIR/output:$ROOT_DIR/output/chapters" \
  --toc \
  --toc-depth=2 \
  --number-sections \
  -V documentclass=article \
  -V CJKmainfont="$FONT" \
  -V mainfont="Times New Roman" \
  -V monofont="$MONO_FONT" \
  -V geometry:margin="$MARGIN" \
  -V colorlinks=true \
  -V linkcolor=blue \
  -V urlcolor=blue

echo "Done: $OUTPUT"
