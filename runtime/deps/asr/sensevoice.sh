#!/bin/sh
set -eu

. /opt/chat2me-deps/lib.sh

pip_install \
  "kaldi_native_fbank" \
  "onnx==1.16.1" \
  "onnxruntime==1.16.3" \
  "sense-voice-streaming-asr==0.1.1" \
  "sentencepiece"

python3 - <<'PY'
from pathlib import Path

import sense_voice_streaming_asr

package_dir = Path(sense_voice_streaming_asr.__file__).resolve().parent

main_path = package_dir / "sense_voice_streaming_asr.py"
if main_path.is_file():
    text = main_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in text.splitlines()[:5]:
        main_path.write_text("from __future__ import annotations\n" + text, encoding="utf-8")

model_data_path = package_dir / "model_data.py"
if model_data_path.is_file():
    text = model_data_path.read_text(encoding="utf-8")
    old = """        with (
            SENSEVOICE_CMVN_PATH as cmvn_path,
            SENSEVOICE_TOKENS_PATH as tokens_json_path,
            SENSEVOICE_MODEL_PATH as model_path,
        ):
"""
    new = """        with SENSEVOICE_CMVN_PATH as cmvn_path, \\
                SENSEVOICE_TOKENS_PATH as tokens_json_path, \\
                SENSEVOICE_MODEL_PATH as model_path:
"""
    if old in text:
        model_data_path.write_text(text.replace(old, new), encoding="utf-8")
PY
