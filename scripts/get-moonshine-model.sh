#!/usr/bin/env bash
# Downloads a Moonshine (moonshine-voice) streaming model from download.moonshine.ai.
#
#   scripts/get-moonshine-model.sh <en|de> <tiny|small|medium> <dest-dir>
#
# The file list and URLs come from the library's model catalog (moonshine_get_stt_catalog).
# Non-English models are under the non-commercial Moonshine Community License
# (https://www.moonshine.ai/license).
set -euo pipefail
LANG_=${1:?language (en|de)}; SIZE=${2:?size (tiny|small|medium)}; DEST=${3:?destination dir}
case "$LANG_-$SIZE" in
  en-tiny|en-small|en-medium) VER=quantized_26_08_21 ;;
  de-tiny|de-small) VER=quantized_26_08_24 ;;
  *) echo "no $SIZE model for '$LANG_'" >&2; exit 1 ;;
esac
BASE="https://download.moonshine.ai/model/$SIZE-streaming-$LANG_/$VER"
mkdir -p "$DEST"
for f in adapter.ort cross_kv.ort decoder_kv.ort encoder.ort frontend.model.ort \
         frontend.weights.ort streaming_config.json tokenizer.bin; do
  [ -s "$DEST/$f" ] && continue
  curl -fsSL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"
done
echo "$DEST"
