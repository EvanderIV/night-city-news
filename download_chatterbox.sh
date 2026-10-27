#!/usr/bin/env bash
# Resilient Chatterbox weight download (resume + retry-all-errors), to dodge
# the flaky HF connection. Mirrors download_model.sh.
set -u

DEST="models/chatterbox"
BASE="https://huggingface.co/ResembleAI/chatterbox/resolve/main"
FILES="tokenizer.json conds.pt ve.safetensors s3gen.safetensors t3_cfg.safetensors"
mkdir -p "$DEST"

for f in $FILES; do
  echo "=== $f ==="
  ok=0
  for attempt in $(seq 1 40); do
    curl -sL -C - --retry 10 --retry-all-errors --retry-delay 3 \
         --connect-timeout 30 \
         -o "$DEST/$f" "$BASE/$f"
    rc=$?
    if [ $rc -eq 0 ]; then ok=1; echo "  ok (attempt $attempt)"; break; fi
    echo "  attempt $attempt failed (curl rc=$rc); resuming..."
    sleep 2
  done
  if [ $ok -ne 1 ]; then echo "FAILED: $f"; exit 1; fi
  echo "  size: $(stat -c %s "$DEST/$f" 2>/dev/null) bytes"
done
echo "ALL CHATTERBOX FILES DOWNLOADED"
