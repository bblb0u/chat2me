#!/bin/sh
set -eu

MODELS_DIR="${MODELS_DIR:-/models}"
DEFAULT_CONFIG_DIR="${DEFAULT_CONFIG_DIR:-/defaults/config}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-$CONFIG_DIR/runtime.env}"

default_voice_model_set() {
  case "$VOICE_ROLE" in
    chat2me-speech) echo "kws" ;;
    chat2me-asr) echo "asr-service" ;;
    chat2me-tts) echo "tts-service" ;;
    *) echo "kws,speech" ;;
  esac
}

model_selected() {
  case ",$MODEL_SET," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

lock_key() {
  printf '%s' "$MODEL_SET" | tr -c 'A-Za-z0-9_.-' '-'
}

normalize_key() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

env_flag_enabled() {
  case "$(normalize_key "${1:-}")" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

known_value_error() {
  name="$1"
  value="$2"
  allowed="$3"
  echo "$name '$value' is not supported. Allowed values: $allowed" >&2
  exit 1
}

resolve_kws_model() {
  KWS_MODEL_NAME="sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
  case "$KWS_MODEL_NAME" in
    sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20)
      KWS_MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"
      ;;
    *)
      known_value_error "KWS_MODEL" "$KWS_MODEL_NAME" "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
      ;;
  esac
}

resolve_asr_model() {
  VOICE_ASR_ENGINE="$(normalize_key "${VOICE_ASR_ENGINE:-sensevoice}")"
  case "$VOICE_ASR_ENGINE" in
    sherpa)
      VOICE_ASR_ENGINE="sherpa"
      VOICE_ASR_MODEL="${VOICE_ASR_MODEL:-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20}"
      case "$VOICE_ASR_MODEL" in
        sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20)
          ASR_MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
          ;;
        *)
          known_value_error "VOICE_ASR_MODEL" "$VOICE_ASR_MODEL" "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
          ;;
      esac
      ;;
    sensevoice)
      VOICE_ASR_MODEL="${VOICE_ASR_MODEL:-SenseVoiceSmall}"
      case "$VOICE_ASR_MODEL" in
        SenseVoiceSmall)
          ASR_HF_REPO_ID="haixuantao/SenseVoiceSmall-onnx"
          ASR_HF_REVISION="main"
          ASR_REQUIRED_FILES="config.yaml,model_quant.onnx,am.mvn,tokens.json"
          VAD_HF_REPO_ID="manyeyes/speech_fsmn_vad_zh-cn-16k-common-onnx"
          VAD_HF_REVISION="main"
          VAD_REQUIRED_FILES="model_quant.onnx,vad.mvn"
          ;;
        *)
          known_value_error "VOICE_ASR_MODEL" "$VOICE_ASR_MODEL" "SenseVoiceSmall"
          ;;
      esac
      ;;
    online)
      VOICE_ASR_MODEL="${VOICE_ASR_MODEL:-gpt-4o-mini-transcribe}"
      ;;
    *)
      known_value_error "VOICE_ASR_ENGINE" "$VOICE_ASR_ENGINE" "sherpa, sensevoice, online"
      ;;
  esac
}

resolve_tts_model() {
  VOICE_TTS_ENGINE="$(normalize_key "${VOICE_TTS_ENGINE:-melotts}")"
  case "$VOICE_TTS_ENGINE" in
    piper)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-zh_CN-huayan-medium}"
      case "$VOICE_TTS_MODEL" in
        zh_CN-huayan-medium)
          PIPER_MODEL_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx"
          PIPER_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"
          ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "zh_CN-huayan-medium"
          ;;
      esac
      ;;
    melotts)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-vits-melo-tts-zh_en}"
      case "$VOICE_TTS_MODEL" in
        vits-melo-tts-zh_en)
          MELOTTS_MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"
          MELOTTS_REQUIRED_FILES="model.onnx,tokens.txt,lexicon.txt,dict/jieba.dict.utf8,phone.fst,date.fst,number.fst,new_heteronym.fst"
          ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "vits-melo-tts-zh_en"
          ;;
      esac
      ;;
    sherpa)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-matcha-icefall-zh-en}"
      case "$VOICE_TTS_MODEL" in
        matcha-icefall-zh-en)
          SHERPA_TTS_MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2"
          SHERPA_TTS_VOCODER_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx"
          SHERPA_TTS_REQUIRED_FILES="model-steps-3.onnx,vocos-16khz-univ.onnx,tokens.txt,lexicon.txt,phone-zh.fst,date-zh.fst,number-zh.fst"
          ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "matcha-icefall-zh-en"
          ;;
      esac
      ;;
    f5-tts)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-F5TTS_v1_Base}"
      case "$VOICE_TTS_MODEL" in
        F5TTS_v1_Base)
          F5_TTS_HF_REPO_ID="SWivid/F5-TTS"
          F5_TTS_HF_REVISION="main"
          F5_TTS_CKPT_REMOTE_FILE="F5TTS_v1_Base/model_1250000.safetensors"
          F5_TTS_VOCAB_REMOTE_FILE="F5TTS_v1_Base/vocab.txt"
          F5_TTS_VOCODER_HF_REPO_ID="charactr/vocos-mel-24khz"
          F5_TTS_VOCODER_HF_REVISION="main"
          F5_TTS_VOCODER_REQUIRED_FILES="config.yaml,pytorch_model.bin"
          ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "F5TTS_v1_Base"
          ;;
      esac
      ;;
    cosyvoice)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-CosyVoice-300M-SFT}"
      case "$VOICE_TTS_MODEL" in
        CosyVoice-300M-SFT)
          COSYVOICE_HF_REPO_ID="FunAudioLLM/CosyVoice-300M-SFT"
          COSYVOICE_HF_REVISION="main"
          COSYVOICE_REQUIRED_FILES="cosyvoice.yaml,flow.pt,hift.pt,llm.pt,spk2info.pt,campplus.onnx,speech_tokenizer_v1.onnx"
          ;;
        CosyVoice-300M-Instruct)
          COSYVOICE_HF_REPO_ID="FunAudioLLM/CosyVoice-300M-Instruct"
          COSYVOICE_HF_REVISION="main"
          COSYVOICE_REQUIRED_FILES="cosyvoice.yaml,flow.pt,hift.pt,llm.pt,spk2info.pt,campplus.onnx,speech_tokenizer_v1.onnx"
          ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "CosyVoice-300M-SFT, CosyVoice-300M-Instruct"
          ;;
      esac
      ;;
    online)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-gpt-4o-mini-tts}"
      ;;
    *)
      known_value_error "VOICE_TTS_ENGINE" "$VOICE_TTS_ENGINE" "piper, melotts, sherpa, f5-tts, cosyvoice, online"
      ;;
  esac
}

load_runtime_env() {
  [ -f "$RUNTIME_CONFIG_PATH" ] || return

  protected_keys="$(mktemp)"
  env | sed 's/=.*//' > "$protected_keys"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      ''|*[!A-Za-z0-9_]*) continue ;;
    esac

    if ! grep -Fxq "$key" "$protected_keys"; then
      export "$key=$value"
    fi
  done < "$RUNTIME_CONFIG_PATH"
  rm -f "$protected_keys"
}

init_config() {
  if [ ! -d "$DEFAULT_CONFIG_DIR" ]; then
    return
  fi

  mkdir -p "$CONFIG_DIR"
  for source_file in "$DEFAULT_CONFIG_DIR"/*; do
    [ -f "$source_file" ] || continue
    target_file="$CONFIG_DIR/$(basename "$source_file")"
    if [ ! -e "$target_file" ]; then
      cp "$source_file" "$target_file"
      echo "Initialized config: $target_file"
    fi
  done
}

required_files_ok() {
  for required_file in "$@"; do
    if [ ! -s "$required_file" ]; then
      echo "Missing or empty model file: $required_file"
      return 1
    fi
  done
}

kws_runtime_ok() {
  python3 - "$KWS_MODEL" "$WAKE_WORDS" <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path

import sherpa_onnx

model_dir = sys.argv[1]
wake_words = [word.strip() for word in sys.argv[2].split(",") if word.strip()]
if not wake_words:
    print("WAKE_WORDS must contain at least one wake word", file=sys.stderr)
    sys.exit(1)

try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_keywords = tmp_path / "keywords_raw.txt"
        keywords = tmp_path / "keywords.txt"
        raw_keywords.write_text("".join(f"{word} @{word}\n" for word in wake_words), encoding="utf-8")
        subprocess.run(
            [
                "sherpa-onnx-cli",
                "text2token",
                "--tokens",
                f"{model_dir}/tokens.txt",
                "--tokens-type",
                "phone+ppinyin",
                "--lexicon",
                f"{model_dir}/en.phone",
                str(raw_keywords),
                str(keywords),
            ],
            check=True,
        )

        sherpa_onnx.KeywordSpotter(
            tokens=f"{model_dir}/tokens.txt",
            encoder=f"{model_dir}/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
            decoder=f"{model_dir}/decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
            joiner=f"{model_dir}/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
            num_threads=1,
            keywords_file=str(keywords),
            provider="cpu",
        )
except Exception as exc:
    print(f"Invalid KWS model: {model_dir}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

asr_runtime_ok() {
  python3 - "$ASR_MODEL" "$AUDIO_SAMPLE_RATE" "${ASR_MODEL_PRECISION:-fp32}" "${ASR_DECODING_METHOD:-modified_beam_search}" "${ASR_MAX_ACTIVE_PATHS:-8}" <<'PY'
import sys
import sherpa_onnx

model_dir = sys.argv[1]
sample_rate = int(sys.argv[2])
precision = sys.argv[3].strip().lower()
decoding_method = sys.argv[4]
max_active_paths = int(sys.argv[5])
suffix = ".int8.onnx" if precision in {"int8", "quantized"} else ".onnx"
try:
    sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder-epoch-99-avg-1{suffix}",
        decoder=f"{model_dir}/decoder-epoch-99-avg-1{suffix}",
        joiner=f"{model_dir}/joiner-epoch-99-avg-1{suffix}",
        num_threads=1,
        sample_rate=sample_rate,
        feature_dim=80,
        enable_endpoint_detection=True,
        decoding_method=decoding_method,
        max_active_paths=max_active_paths,
        provider="cpu",
    )
except Exception as exc:
    print(f"Invalid ASR model: {model_dir}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

kws_model_ok() {
  required_files_ok \
    "$KWS_MODEL/tokens.txt" \
    "$KWS_MODEL/en.phone" \
    "$KWS_MODEL/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx" \
    "$KWS_MODEL/decoder-epoch-13-avg-2-chunk-8-left-64.onnx" \
    "$KWS_MODEL/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx" \
    && kws_runtime_ok
}

asr_model_ok() {
  case "${ASR_MODEL_PRECISION:-fp32}" in
    int8|quantized) asr_suffix=".int8.onnx" ;;
    fp32|float32|full) asr_suffix=".onnx" ;;
    *) echo "ASR_MODEL_PRECISION must be fp32 or int8" >&2; return 1 ;;
  esac
  required_files_ok \
    "$ASR_MODEL/tokens.txt" \
    "$ASR_MODEL/encoder-epoch-99-avg-1$asr_suffix" \
    "$ASR_MODEL/decoder-epoch-99-avg-1$asr_suffix" \
    "$ASR_MODEL/joiner-epoch-99-avg-1$asr_suffix" \
    && asr_runtime_ok
}

sherpa_tts_runtime_ok() {
  python_module_ok sherpa_onnx
}

sherpa_tts_model_ok() {
  required_relative_files_ok "$TTS_MODEL_DIR" "$SHERPA_TTS_REQUIRED_FILES" \
    && [ -d "$TTS_MODEL_DIR/espeak-ng-data" ] \
    && sherpa_tts_runtime_ok
}

melotts_runtime_ok() {
  python_module_ok sherpa_onnx
}

melotts_model_ok() {
  required_relative_files_ok "$TTS_MODEL_DIR" "$MELOTTS_REQUIRED_FILES" \
    && melotts_runtime_ok
}

piper_runtime_ok() {
  command -v piper >/dev/null 2>&1 \
    && [ -d "${PIPER_ESPEAK_DATA:-/opt/piper/espeak-ng-data}" ]
}

piper_model_ok() {
  required_files_ok \
    "$TTS_MODEL_DIR/model.onnx" \
    "$TTS_MODEL_DIR/model.onnx.json" \
    && piper_runtime_ok
}

cosyvoice_model_ok() {
  required_relative_files_ok "$TTS_MODEL_DIR" "$COSYVOICE_REQUIRED_FILES"
}

dir_has_files() {
  dir="$1"
  [ -d "$dir" ] || return 1
  find "$dir" -type f -print -quit | grep -q .
}

required_relative_files_ok() {
  base_dir="$1"
  required_files="$2"

  if [ -z "$required_files" ]; then
    dir_has_files "$base_dir"
    return
  fi

  old_ifs="$IFS"
  IFS=","
  set -- $required_files
  IFS="$old_ifs"
  for relative_file in "$@"; do
    relative_file="$(printf '%s' "$relative_file" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -n "$relative_file" ] || continue
    required_files_ok "$base_dir/$relative_file" || return 1
  done
}

sensevoice_model_ok() {
  sensevoice_runtime_ok \
    && required_relative_files_ok "$ASR_MODEL" "$ASR_REQUIRED_FILES" \
    && sensevoice_vad_files_ok
}

sensevoice_asr_model_ok() {
  sensevoice_runtime_ok \
    && required_relative_files_ok "$ASR_MODEL" "$ASR_REQUIRED_FILES"
}

sensevoice_vad_model_ok() {
  sensevoice_runtime_ok \
    && sensevoice_vad_files_ok
}

sensevoice_vad_files_ok() {
  required_relative_files_ok "$VAD_MODEL_DIR" "$VAD_REQUIRED_FILES" || return 1
  if [ -s "$VAD_MODEL_DIR/vad.mvn" ] || [ -s "$VAD_MODEL_DIR/am.mvn" ]; then
    return 0
  fi
  echo "Missing VAD CMVN file: $VAD_MODEL_DIR/vad.mvn or $VAD_MODEL_DIR/am.mvn"
  return 1
}

python_module_ok() {
  python3 - "$1" <<'PY'
import importlib.util
import sys

try:
    found = importlib.util.find_spec(sys.argv[1]) is not None
except ModuleNotFoundError:
    found = False
sys.exit(0 if found else 1)
PY
}

missing_image_dependency() {
  echo "$1 is missing from the image. Rebuild the Docker image with the required dependency baked in." >&2
  exit 1
}

require_python_module() {
  module="$1"
  python_module_ok "$module" || missing_image_dependency "Python module '$module'"
}

require_command() {
  command_name="$1"
  command -v "$command_name" >/dev/null 2>&1 || missing_image_dependency "Command '$command_name'"
}

verify_torch_cuda_runtime() {
  python3 <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print(f"torch cuda unavailable: torch={torch.__version__} cuda={getattr(torch.version, 'cuda', None)}", file=sys.stderr)
    sys.exit(1)

print(f"[runtime] torch cuda ready: torch={torch.__version__} cuda={torch.version.cuda} devices={torch.cuda.device_count()}")
PY
}

ensure_kws_runtime() {
  require_python_module sherpa_onnx
  require_command sherpa-onnx-cli
}

ensure_sherpa_asr_runtime() {
  ensure_kws_runtime
}

ensure_sherpa_tts_runtime() {
  if sherpa_tts_runtime_ok; then
    return
  fi
  missing_image_dependency "Sherpa ONNX TTS runtime"
}

ensure_melotts_runtime() {
  if melotts_runtime_ok; then
    return
  fi
  missing_image_dependency "MeloTTS ONNX runtime"
}

ensure_piper_runtime() {
  if piper_runtime_ok; then
    return
  fi
  missing_image_dependency "Piper runtime"
}

sensevoice_runtime_ok() {
  python_module_ok sense_voice_streaming_asr \
    && python_module_ok onnxruntime \
    && python_module_ok kaldi_native_fbank \
    && python_module_ok sentencepiece
}

ensure_sensevoice_runtime() {
  if sensevoice_runtime_ok; then
    return
  fi
  missing_image_dependency "SenseVoice streaming ASR runtime"
}

cosyvoice_runtime_ok() {
  python_module_ok torch \
    && python_module_ok onnxruntime \
    && python_module_ok hyperpyyaml \
    && python_module_ok transformers \
    && python_module_ok librosa \
    && python_module_ok scipy \
    && python_module_ok cosyvoice.cli.cosyvoice \
    && python_module_ok matcha
}

ensure_cosyvoice_runtime() {
  COSYVOICE_CODE_DIR="${COSYVOICE_CODE_DIR:-/opt/CosyVoice}"
  export COSYVOICE_PACKAGE_PATH="${COSYVOICE_PACKAGE_PATH:-$COSYVOICE_CODE_DIR:$COSYVOICE_CODE_DIR/third_party/Matcha-TTS}"
  export PYTHONPATH="$COSYVOICE_PACKAGE_PATH${PYTHONPATH:+:$PYTHONPATH}"
  [ -d "$COSYVOICE_CODE_DIR/cosyvoice" ] || missing_image_dependency "CosyVoice code directory '$COSYVOICE_CODE_DIR/cosyvoice'"
  [ -d "$COSYVOICE_CODE_DIR/third_party/Matcha-TTS/matcha" ] || missing_image_dependency "Matcha-TTS code directory '$COSYVOICE_CODE_DIR/third_party/Matcha-TTS/matcha'"
  if ! cosyvoice_runtime_ok; then
    missing_image_dependency "CosyVoice runtime"
  fi
  case "$(normalize_key "${VOICE_TTS_DEVICE:-auto}")" in
    auto|cuda|gpu|cuda:*) verify_torch_cuda_runtime ;;
    *)
      echo "CosyVoice requires GPU. Set VOICE_TTS_DEVICE=cuda or auto." >&2
      exit 1
      ;;
  esac
}

ensure_online_asr_runtime() {
  require_python_module httpx
  require_command ffmpeg
}

ensure_online_tts_runtime() {
  require_python_module httpx
  require_command ffmpeg
}

f5_tts_runtime_ok() {
  python_module_ok torch \
    && python_module_ok f5_tts \
    && python_module_ok vocos \
    && python_module_ok safetensors \
    && python_module_ok soundfile \
    && python_module_ok scipy \
    && python_module_ok pypinyin \
    && python_module_ok rjieba \
    && python_module_ok x_transformers \
    && python_module_ok torchdiffeq \
    && python_module_ok einops \
    && python_module_ok librosa \
    && python_module_ok importlib_resources
}

ensure_f5_tts_runtime() {
  if ! f5_tts_runtime_ok; then
    missing_image_dependency "F5-TTS runtime"
  fi
  case "$(normalize_key "${VOICE_TTS_DEVICE:-auto}")" in
    cuda|gpu|cuda:*) verify_torch_cuda_runtime ;;
  esac
}

ensure_selected_runtimes() {
  if model_selected kws; then
    ensure_kws_runtime
  fi

  if model_selected speech || model_selected asr-service; then
    case "$VOICE_ASR_ENGINE" in
      sherpa) ensure_sherpa_asr_runtime ;;
      sensevoice) ensure_sensevoice_runtime ;;
      online) ensure_online_asr_runtime ;;
    esac
    if [ "$VOICE_ASR_ENGINE" = "online" ]; then
      case "${VOICE_ASR_FALLBACK_ENGINE:-sensevoice}" in
        sherpa) ensure_sherpa_asr_runtime ;;
        sensevoice) ensure_sensevoice_runtime ;;
      esac
    fi
  fi

  if model_selected speech || model_selected tts-service; then
    case "$VOICE_TTS_ENGINE" in
      piper) ensure_piper_runtime ;;
      melotts) ensure_melotts_runtime ;;
      sherpa) ensure_sherpa_tts_runtime ;;
      f5-tts) ensure_f5_tts_runtime ;;
      cosyvoice) ensure_cosyvoice_runtime ;;
      online) ensure_online_tts_runtime ;;
    esac
    if [ "$VOICE_TTS_ENGINE" = "online" ]; then
      case "${VOICE_TTS_FALLBACK_ENGINE:-melotts}" in
        piper) ensure_piper_runtime ;;
        melotts) ensure_melotts_runtime ;;
        sherpa) ensure_sherpa_tts_runtime ;;
        f5-tts) ensure_f5_tts_runtime ;;
        cosyvoice) ensure_cosyvoice_runtime ;;
      esac
    fi
  fi
}

content_length() {
  curl -fsSIL --retry 5 --connect-timeout 20 "$1" \
    | awk 'tolower($1) == "content-length:" { size = $2 } END { gsub("\r", "", size); print size }'
}

print_download_progress() {
  label="$1"
  completed="$2"
  total="$3"

  case "$completed" in ''|*[!0-9]*) completed=0 ;; esac
  case "$total" in ''|*[!0-9]*) total=0 ;; esac

  if [ "$total" -gt 0 ]; then
    awk -v label="$label" -v done="$completed" -v total="$total" '
      BEGIN {
        width = 24
        pct = int(done * 100 / total)
        if (pct > 100) pct = 100
        filled = int(pct * width / 100)
        bar = ""
        for (i = 0; i < width; i++) bar = bar (i < filled ? "#" : "-")
        printf("[models] %s [%s] %3d%% %.1f/%.1f MB\n", label, bar, pct, done / 1048576, total / 1048576)
      }'
  else
    awk -v label="$label" -v done="$completed" '
      BEGIN {
        printf("[models] %s %.1f MB downloaded\n", label, done / 1048576)
      }'
  fi
}

download_with_progress() {
  output="$1"
  url="$2"
  label="$3"
  tmp="$output.download"
  max_attempts="${MODEL_DOWNLOAD_RETRIES:-10}"
  attempt=1

  mkdir -p "$(dirname "$output")"
  total="$(content_length "$url" || true)"
  echo "[models] downloading $label"

  while [ "$attempt" -le "$max_attempts" ]; do
    if [ -f "$tmp" ]; then
      completed="$(wc -c < "$tmp" | tr -d ' ')"
      echo "[models] resuming $label from $(awk -v done="$completed" 'BEGIN { printf("%.1f", done / 1048576) }') MB"
    else
      completed=0
    fi
    print_download_progress "$label" "$completed" "$total"

    curl -fL --retry 10 --retry-connrefused --connect-timeout 20 --speed-limit 1024 --speed-time 120 --continue-at - --silent --show-error "$url" -o "$tmp" &
    curl_pid="$!"

    (
      while kill -0 "$curl_pid" 2>/dev/null; do
        if [ -f "$tmp" ]; then
          completed="$(wc -c < "$tmp" | tr -d ' ')"
          print_download_progress "$label" "$completed" "$total"
        fi
        sleep 5
      done
    ) &
    progress_pid="$!"

    if wait "$curl_pid"; then
      curl_ok=1
    else
      curl_ok=0
    fi

    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true

    if [ "$curl_ok" -eq 1 ]; then
      completed="$(wc -c < "$tmp" | tr -d ' ')"
      print_download_progress "$label" "$completed" "$total"
      if [ "${total:-0}" -gt 0 ] && [ "$completed" -lt "$total" ]; then
        echo "[models] incomplete download for $label: $completed/$total bytes" >&2
      else
        mv "$tmp" "$output"
        return 0
      fi
    fi

    if [ "$attempt" -ge "$max_attempts" ]; then
      rm -f "$tmp"
      return 1
    fi
    echo "[models] retrying $label download ($attempt/$max_attempts)"
    attempt=$((attempt + 1))
    sleep 5
  done

  rm -f "$tmp"
  return 1
}

download_and_extract() {
  name="$1"
  url="$2"
  target="$3"
  archive="$MODELS_DIR/$name.tar.bz2"

  echo "[models] preparing $name"
  rm -rf "$target"
  download_with_progress "$archive" "$url" "$name"
  echo "[models] extracting $name"
  tmp_extract_dir="$MODELS_DIR/.extract.$name"
  rm -rf "$tmp_extract_dir"
  mkdir -p "$tmp_extract_dir"
  python3 - "$archive" "$tmp_extract_dir" <<'PY'
import sys
import tarfile

with tarfile.open(sys.argv[1], "r:bz2") as archive:
    archive.extractall(sys.argv[2])
PY
  extracted_dir="$tmp_extract_dir/$name"
  if [ -d "$extracted_dir" ]; then
    mkdir -p "$(dirname "$target")"
    mv "$extracted_dir" "$target"
  else
    mkdir -p "$target"
    find "$tmp_extract_dir" -mindepth 1 -maxdepth 1 -exec mv {} "$target"/ \;
  fi
  rm -rf "$tmp_extract_dir"
  rm -f "$archive"
  echo "[models] extracted $name"
}

download_hf_snapshot() {
  target="$1"
  repo_id="$2"
  revision="$3"
  label="$4"
  required_files="$5"

  rm -rf "$target"
  mkdir -p "$target"
  echo "[models] downloading $label from Hugging Face repo $repo_id"
  old_ifs="$IFS"
  IFS=","
  set -- $required_files
  IFS="$old_ifs"
  for relative_file in "$@"; do
    relative_file="$(printf '%s' "$relative_file" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    [ -n "$relative_file" ] || continue
    download_hf_file "$target/$relative_file" "$repo_id" "$revision" "$relative_file"
  done
}

download_hf_file() {
  output="$1"
  repo_id="$2"
  revision="$3"
  relative_file="$4"

  endpoints=""
  if [ -n "${HF_ENDPOINT:-}" ]; then
    endpoints="$HF_ENDPOINT"
  fi
  endpoints="$endpoints https://hf-mirror.com https://huggingface.co"

  for endpoint in $endpoints; do
    endpoint="${endpoint%/}"
    url="$endpoint/$repo_id/resolve/${revision:-main}/$relative_file"
    if download_with_progress "$output" "$url" "$relative_file"; then
      return 0
    fi
    echo "[models] failed downloading $repo_id/$relative_file via $endpoint" >&2
  done

  echo "failed to download $repo_id/$relative_file" >&2
  return 1
}

ensure_archive_model() {
  name="$1"
  url="$2"
  check_name="$3"
  target="$4"

  if "$check_name"; then
    echo "$name is ready"
    return
  fi

  echo "$name is missing or invalid; re-downloading"
  download_and_extract "$name" "$url" "$target"

  echo "[models] validating $name"
  if ! "$check_name"; then
    echo "$name is still invalid after download" >&2
    exit 1
  fi
}

ensure_hf_snapshot_model() {
  name="$1"
  repo_id="$2"
  revision="$3"
  check_name="$4"
  target="$5"
  required_files="$6"

  if "$check_name"; then
    echo "$name is ready"
    return
  fi

  echo "$name is missing or invalid; re-downloading"
  download_hf_snapshot "$target" "$repo_id" "$revision" "$name" "$required_files"

  echo "[models] validating $name"
  if ! "$check_name"; then
    echo "$name is still invalid after download" >&2
    exit 1
  fi
}

ensure_sherpa_tts_model() {
  if sherpa_tts_model_ok; then
    echo "sherpa $VOICE_TTS_MODEL is ready"
    return
  fi

  echo "sherpa $VOICE_TTS_MODEL is missing or invalid; re-downloading"
  download_and_extract "$VOICE_TTS_MODEL" "$SHERPA_TTS_MODEL_URL" "$TTS_MODEL_DIR"
  download_with_progress "$TTS_MODEL_DIR/vocos-16khz-univ.onnx" "$SHERPA_TTS_VOCODER_URL" "vocos-16khz-univ.onnx"

  echo "[models] validating sherpa $VOICE_TTS_MODEL"
  if ! sherpa_tts_model_ok; then
    echo "sherpa $VOICE_TTS_MODEL is still invalid after download" >&2
    exit 1
  fi
}

ensure_melotts_model() {
  if melotts_model_ok; then
    echo "melotts $VOICE_TTS_MODEL is ready"
    return
  fi

  echo "melotts $VOICE_TTS_MODEL is missing or invalid; re-downloading"
  download_and_extract "$VOICE_TTS_MODEL" "$MELOTTS_MODEL_URL" "$TTS_MODEL_DIR"

  echo "[models] validating melotts $VOICE_TTS_MODEL"
  if ! melotts_model_ok; then
    echo "melotts $VOICE_TTS_MODEL is still invalid after download" >&2
    exit 1
  fi
}

ensure_piper_model() {
  if piper_model_ok; then
    echo "piper $VOICE_TTS_MODEL is ready"
    return
  fi

  echo "piper $VOICE_TTS_MODEL is missing or invalid; re-downloading"
  rm -rf "$TTS_MODEL_DIR"
  download_with_progress "$TTS_MODEL_DIR/model.onnx" "$PIPER_MODEL_URL" "$VOICE_TTS_MODEL.onnx"
  download_with_progress "$TTS_MODEL_DIR/model.onnx.json" "$PIPER_CONFIG_URL" "$VOICE_TTS_MODEL.onnx.json"

  echo "[models] validating piper $VOICE_TTS_MODEL"
  if ! piper_model_ok; then
    echo "piper $VOICE_TTS_MODEL is still invalid after download" >&2
    exit 1
  fi
}

ensure_cosyvoice_model() {
  ensure_hf_snapshot_model \
    "$VOICE_TTS_MODEL" \
    "$COSYVOICE_HF_REPO_ID" \
    "$COSYVOICE_HF_REVISION" \
    cosyvoice_model_ok \
    "$TTS_MODEL_DIR" \
    "$COSYVOICE_REQUIRED_FILES"
}

f5_tts_model_ok() {
  required_files_ok \
    "$TTS_MODEL_DIR/model_1250000.safetensors" \
    "$TTS_MODEL_DIR/vocab.txt" \
    "$TTS_MODEL_DIR/config.yaml" \
    "$MODELS_DIR/f5-tts/vocos-mel-24khz/config.yaml" \
    "$MODELS_DIR/f5-tts/vocos-mel-24khz/pytorch_model.bin"
}

ensure_f5_tts_model() {
  if f5_tts_model_ok; then
    echo "f5-tts $VOICE_TTS_MODEL is ready"
    return
  fi

  echo "f5-tts $VOICE_TTS_MODEL is missing or invalid; re-downloading"
  rm -rf "$TTS_MODEL_DIR"
  mkdir -p "$TTS_MODEL_DIR"
  download_hf_file \
    "$TTS_MODEL_DIR/model_1250000.safetensors" \
    "$F5_TTS_HF_REPO_ID" \
    "$F5_TTS_HF_REVISION" \
    "$F5_TTS_CKPT_REMOTE_FILE"
  download_hf_file \
    "$TTS_MODEL_DIR/vocab.txt" \
    "$F5_TTS_HF_REPO_ID" \
    "$F5_TTS_HF_REVISION" \
    "$F5_TTS_VOCAB_REMOTE_FILE"
  cp "/usr/local/lib/python3.8/dist-packages/f5_tts/configs/$VOICE_TTS_MODEL.yaml" "$TTS_MODEL_DIR/config.yaml"

  ensure_hf_snapshot_model \
    "vocos-mel-24khz" \
    "$F5_TTS_VOCODER_HF_REPO_ID" \
    "$F5_TTS_VOCODER_HF_REVISION" \
    f5_tts_vocoder_files_ok \
    "$MODELS_DIR/f5-tts/vocos-mel-24khz" \
    "$F5_TTS_VOCODER_REQUIRED_FILES"

  echo "[models] validating f5-tts $VOICE_TTS_MODEL"
  if ! f5_tts_model_ok; then
    echo "f5-tts $VOICE_TTS_MODEL is still invalid after download" >&2
    exit 1
  fi
}

f5_tts_checkpoint_files_ok() {
  required_files_ok \
    "$TTS_MODEL_DIR/model_1250000.safetensors" \
    "$TTS_MODEL_DIR/vocab.txt"
}

f5_tts_vocoder_files_ok() {
  required_files_ok \
    "$MODELS_DIR/f5-tts/vocos-mel-24khz/config.yaml" \
    "$MODELS_DIR/f5-tts/vocos-mel-24khz/pytorch_model.bin"
}

init_config
load_runtime_env

VOICE_MODELS_REQUIRED="${VOICE_MODELS_REQUIRED:-1}"
VOICE_ROLE="${VOICE_ROLE:-}"
resolve_kws_model
resolve_asr_model
resolve_tts_model
: "${WAKE_WORDS:?WAKE_WORDS must be set in runtime.env}"
: "${AUDIO_SAMPLE_RATE:?AUDIO_SAMPLE_RATE must be set in runtime.env}"
KWS_MODEL="$MODELS_DIR/$KWS_MODEL_NAME"
ASR_MODEL="$MODELS_DIR/$VOICE_ASR_ENGINE/$VOICE_ASR_MODEL"
VAD_MODEL_DIR="$MODELS_DIR/$VOICE_ASR_ENGINE/speech_fsmn_vad_zh-cn-16k-common-onnx"
TTS_MODEL_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
export VOICE_ASR_ENGINE
export VOICE_ASR_MODEL
export VOICE_TTS_ENGINE
export VOICE_TTS_MODEL
export SENSEVOICE_MODEL_DIR="$ASR_MODEL"
export SENSEVOICE_VAD_MODEL_DIR="$VAD_MODEL_DIR"
ORIGINAL_VOICE_ASR_ENGINE="$VOICE_ASR_ENGINE"
ORIGINAL_VOICE_ASR_MODEL="$VOICE_ASR_MODEL"
ORIGINAL_VOICE_TTS_ENGINE="$VOICE_TTS_ENGINE"
ORIGINAL_VOICE_TTS_MODEL="$VOICE_TTS_MODEL"

prepare_asr_download_target() {
  if [ "$ORIGINAL_VOICE_ASR_ENGINE" = "online" ]; then
    VOICE_ASR_ENGINE="${VOICE_ASR_FALLBACK_ENGINE:-sensevoice}"
    VOICE_ASR_MODEL="${VOICE_ASR_FALLBACK_MODEL:-SenseVoiceSmall}"
  else
    VOICE_ASR_ENGINE="$ORIGINAL_VOICE_ASR_ENGINE"
    VOICE_ASR_MODEL="$ORIGINAL_VOICE_ASR_MODEL"
  fi
  resolve_asr_model
  ASR_MODEL="$MODELS_DIR/$VOICE_ASR_ENGINE/$VOICE_ASR_MODEL"
  VAD_MODEL_DIR="$MODELS_DIR/$VOICE_ASR_ENGINE/speech_fsmn_vad_zh-cn-16k-common-onnx"
  export SENSEVOICE_MODEL_DIR="$ASR_MODEL"
  export SENSEVOICE_VAD_MODEL_DIR="$VAD_MODEL_DIR"
}

prepare_tts_download_target() {
  if [ "$ORIGINAL_VOICE_TTS_ENGINE" = "online" ]; then
    VOICE_TTS_ENGINE="${VOICE_TTS_FALLBACK_ENGINE:-melotts}"
    VOICE_TTS_MODEL="${VOICE_TTS_FALLBACK_MODEL:-vits-melo-tts-zh_en}"
  else
    VOICE_TTS_ENGINE="$ORIGINAL_VOICE_TTS_ENGINE"
    VOICE_TTS_MODEL="$ORIGINAL_VOICE_TTS_MODEL"
  fi
  resolve_tts_model
  TTS_MODEL_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
}

restore_runtime_model_selection() {
  VOICE_ASR_ENGINE="$ORIGINAL_VOICE_ASR_ENGINE"
  VOICE_ASR_MODEL="$ORIGINAL_VOICE_ASR_MODEL"
  VOICE_TTS_ENGINE="$ORIGINAL_VOICE_TTS_ENGINE"
  VOICE_TTS_MODEL="$ORIGINAL_VOICE_TTS_MODEL"
  resolve_asr_model
  resolve_tts_model
  ASR_MODEL="$MODELS_DIR/$VOICE_ASR_ENGINE/$VOICE_ASR_MODEL"
  VAD_MODEL_DIR="$MODELS_DIR/$VOICE_ASR_ENGINE/speech_fsmn_vad_zh-cn-16k-common-onnx"
  TTS_MODEL_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
  export VOICE_ASR_ENGINE
  export VOICE_ASR_MODEL
  export VOICE_TTS_ENGINE
  export VOICE_TTS_MODEL
  export SENSEVOICE_MODEL_DIR="$ASR_MODEL"
  export SENSEVOICE_VAD_MODEL_DIR="$VAD_MODEL_DIR"
}

MODEL_SET="$(default_voice_model_set)"
if model_selected speech || model_selected asr-service; then
  case "$VOICE_ASR_ENGINE" in
    sherpa) MODEL_SET="$MODEL_SET,asr" ;;
    sensevoice) MODEL_SET="$MODEL_SET,sensevoice" ;;
    online) ;;
  esac
  if [ "$VOICE_ASR_ENGINE" = "online" ]; then
    VOICE_ASR_FALLBACK_ENGINE="${VOICE_ASR_FALLBACK_ENGINE:-sensevoice}"
    VOICE_ASR_FALLBACK_MODEL="${VOICE_ASR_FALLBACK_MODEL:-SenseVoiceSmall}"
    case "$VOICE_ASR_FALLBACK_ENGINE" in
      sherpa)
        saved_engine="$VOICE_ASR_ENGINE"
        saved_model="$VOICE_ASR_MODEL"
        VOICE_ASR_ENGINE="$VOICE_ASR_FALLBACK_ENGINE"
        VOICE_ASR_MODEL="$VOICE_ASR_FALLBACK_MODEL"
        resolve_asr_model
        MODEL_SET="$MODEL_SET,asr"
        VOICE_ASR_ENGINE="$saved_engine"
        VOICE_ASR_MODEL="$saved_model"
        resolve_asr_model
        ;;
      sensevoice) MODEL_SET="$MODEL_SET,sensevoice" ;;
      *) echo "VOICE_ASR_FALLBACK_ENGINE '$VOICE_ASR_FALLBACK_ENGINE' is not supported" >&2; exit 1 ;;
    esac
  fi
fi
if model_selected speech || model_selected tts-service; then
  case "$VOICE_TTS_ENGINE" in
    piper) MODEL_SET="$MODEL_SET,piper" ;;
    melotts) MODEL_SET="$MODEL_SET,melotts" ;;
    sherpa) MODEL_SET="$MODEL_SET,sherpa-tts" ;;
    f5-tts) MODEL_SET="$MODEL_SET,f5-tts" ;;
    cosyvoice) MODEL_SET="$MODEL_SET,cosyvoice" ;;
    online) ;;
  esac
  if [ "$VOICE_TTS_ENGINE" = "online" ]; then
    VOICE_TTS_FALLBACK_ENGINE="${VOICE_TTS_FALLBACK_ENGINE:-melotts}"
    VOICE_TTS_FALLBACK_MODEL="${VOICE_TTS_FALLBACK_MODEL:-vits-melo-tts-zh_en}"
    case "$VOICE_TTS_FALLBACK_ENGINE" in
      piper) MODEL_SET="$MODEL_SET,piper" ;;
      melotts) MODEL_SET="$MODEL_SET,melotts" ;;
      sherpa) MODEL_SET="$MODEL_SET,sherpa-tts" ;;
      f5-tts) MODEL_SET="$MODEL_SET,f5-tts" ;;
      cosyvoice) MODEL_SET="$MODEL_SET,cosyvoice" ;;
      *) echo "VOICE_TTS_FALLBACK_ENGINE '$VOICE_TTS_FALLBACK_ENGINE' is not supported" >&2; exit 1 ;;
    esac
  fi
fi
: "${LOCK_WAIT_LOG_SECONDS:?LOCK_WAIT_LOG_SECONDS must be set in runtime.env}"

if [ "$VOICE_MODELS_REQUIRED" != "1" ]; then
  exec "$@"
fi

mkdir -p "$MODELS_DIR"
LOCK_DIR="$MODELS_DIR/.download.$(lock_key).lock"
lock_waited=0
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
  sleep 2
  lock_waited=$((lock_waited + 2))
  if [ "$lock_waited" -eq 2 ]; then
    echo "[models] waiting for voice model download lock: $MODEL_SET"
  elif [ "$LOCK_WAIT_LOG_SECONDS" -gt 0 ] && [ $((lock_waited % LOCK_WAIT_LOG_SECONDS)) -eq 0 ]; then
    echo "[models] still waiting for voice model download lock: $MODEL_SET (${lock_waited}s)"
  fi
  if [ "$lock_waited" -ge 600 ]; then
    echo "[models] removing stale voice model download lock after ${lock_waited}s: $LOCK_DIR" >&2
    rmdir "$LOCK_DIR" 2>/dev/null || true
    lock_waited=0
  fi
done
echo "[models] voice model download lock acquired: $MODEL_SET"
trap 'rmdir "$LOCK_DIR"' EXIT

ensure_selected_runtimes

if model_selected kws; then
  ensure_archive_model \
    "$KWS_MODEL_NAME" \
    "$KWS_MODEL_URL" \
    kws_model_ok \
    "$KWS_MODEL"
fi

if model_selected asr; then
  prepare_asr_download_target
  ensure_archive_model \
    "$VOICE_ASR_MODEL" \
    "$ASR_MODEL_URL" \
    asr_model_ok \
    "$ASR_MODEL"
fi

if model_selected sherpa-tts; then
  prepare_tts_download_target
  ensure_sherpa_tts_model
fi

if model_selected piper; then
  prepare_tts_download_target
  ensure_piper_model
fi

if model_selected melotts; then
  prepare_tts_download_target
  ensure_melotts_model
fi

if model_selected sensevoice; then
  prepare_asr_download_target
  ensure_hf_snapshot_model \
    "$VOICE_ASR_MODEL" \
    "$ASR_HF_REPO_ID" \
    "$ASR_HF_REVISION" \
    sensevoice_asr_model_ok \
    "$ASR_MODEL" \
    "$ASR_REQUIRED_FILES"
  ensure_hf_snapshot_model \
    "speech_fsmn_vad_zh-cn-16k-common-onnx" \
    "$VAD_HF_REPO_ID" \
    "$VAD_HF_REVISION" \
    sensevoice_vad_model_ok \
    "$VAD_MODEL_DIR" \
    "$VAD_REQUIRED_FILES"
  if ! sensevoice_model_ok; then
    echo "sensevoice $VOICE_ASR_MODEL is still invalid after download" >&2
    exit 1
  fi
fi

if model_selected f5-tts; then
  prepare_tts_download_target
  ensure_f5_tts_model
fi

if model_selected cosyvoice; then
  prepare_tts_download_target
  ensure_cosyvoice_model
fi

trap - EXIT
rmdir "$LOCK_DIR"

restore_runtime_model_selection
exec "$@"
