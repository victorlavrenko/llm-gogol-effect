#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
DIST="$ROOT/dist"
WORK="$ROOT/.iclr_build"
PKG="$WORK/iclr2027_supplement"

rm -rf "$DIST" "$WORK"
mkdir -p "$DIST" "$PKG"

# 1) Build the anonymous ICLR paper.
(
  cd "$ROOT/paper"
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -jobname=iclr2027_submission iclr_main.tex
  cp iclr2027_submission.pdf "$DIST/iclr2027_submission.pdf"
)

# 2) Preflight PDF anonymity and size.
# Avoid generic first-name checks because they can legitimately occur in research data.
PATTERN='Victor[[:space:]]+Lavrenko|Lavrenko|PeaceTech|peacetech\.vc|victorlavrenko|github\.com/victorlavrenko|C:\\Users\\lavre|/Users/lavre|/home/lavre'
pdftotext "$DIST/iclr2027_submission.pdf" "$WORK/paper.txt"
pdfinfo "$DIST/iclr2027_submission.pdf" > "$WORK/pdfinfo.txt"
if grep -Eiq "$PATTERN" "$WORK/paper.txt" "$WORK/pdfinfo.txt"; then
  echo "ERROR: identifying text found in submission PDF or metadata" >&2
  grep -Ein "$PATTERN" "$WORK/paper.txt" "$WORK/pdfinfo.txt" >&2 || true
  exit 1
fi
PDF_BYTES=$(wc -c < "$DIST/iclr2027_submission.pdf" | tr -d ' ')
PDF_LIMIT=$((50 * 1024 * 1024))
if [ "$PDF_BYTES" -gt "$PDF_LIMIT" ]; then
  echo "ERROR: PDF exceeds OpenReview 50 MB limit" >&2
  exit 1
fi

# 3) Assemble a self-contained anonymous reviewer supplement.
cp -a "$ROOT/reproduction" "$PKG/"
cp -a "$ROOT/gogol_extension" "$PKG/"
cp -a "$ROOT/results" "$PKG/"
cp "$ROOT/reproduce.sh" "$PKG/reproduce.sh"
cp "$ROOT/submission/ANONYMOUS_SUPPLEMENT_README.md" "$PKG/README.md"
chmod +x "$PKG/reproduce.sh"

# Remove transient caches if any are present.
find "$PKG" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$PKG" -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

# 4) Fail closed on author-identifying content in anything uploaded to reviewers.
# Use -a so SQLite/JSON/CSV/log artifacts are also scanned as byte streams.
BAD_LIST="$WORK/identifying_files.txt"
grep -aRIlE "$PATTERN" "$PKG" > "$BAD_LIST" 2>/dev/null || true
if [ -s "$BAD_LIST" ]; then
  echo "ERROR: identifying content found in anonymous supplement:" >&2
  sed 's/^/  /' "$BAD_LIST" >&2
  exit 1
fi

# 5) Deterministic checksums and archive.
(
  cd "$PKG"
  find . -type f ! -name CHECKSUMS.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > CHECKSUMS.sha256
)
(
  cd "$WORK"
  zip -q -r "$DIST/iclr2027_supplement.zip" iclr2027_supplement
)

ZIP_BYTES=$(wc -c < "$DIST/iclr2027_supplement.zip" | tr -d ' ')
ZIP_LIMIT=$((100 * 1024 * 1024))
if [ "$ZIP_BYTES" -gt "$ZIP_LIMIT" ]; then
  echo "ERROR: supplement exceeds OpenReview 100 MB limit" >&2
  exit 1
fi

printf 'Built:\n  %s (%s bytes)\n  %s (%s bytes)\n' \
  "$DIST/iclr2027_submission.pdf" "$PDF_BYTES" \
  "$DIST/iclr2027_supplement.zip" "$ZIP_BYTES"
