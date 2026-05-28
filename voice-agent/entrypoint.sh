#!/bin/sh
set -eu

MODELS_DIR="${MODELS_DIR:-/models}"
DEFAULT_CONFIG_DIR="${DEFAULT_CONFIG_DIR:-/defaults/config}"
CONFIG_DIR="${CONFIG_DIR:-/app/config}"
RUNTIME_CONFIG_PATH="${RUNTIME_CONFIG_PATH:-$CONFIG_DIR/runtime.env}"
PYTHON_RUNTIME_PREFIX="${PYTHON_RUNTIME_PREFIX:-$MODELS_DIR/runtime/python}"
PYTHON_RUNTIME_VERSION="$(python3 - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
PYTHON_RUNTIME_SITE="$PYTHON_RUNTIME_PREFIX/lib/$PYTHON_RUNTIME_VERSION/site-packages"
export PYTHONUSERBASE="$PYTHON_RUNTIME_PREFIX"
export PATH="$PYTHON_RUNTIME_PREFIX/bin:$PATH"
export PYTHONPATH="$PYTHON_RUNTIME_SITE${PYTHONPATH:+:$PYTHONPATH}"

default_voice_model_set() {
  case "$VOICE_ROLE" in
    chat2m-wake) echo "kws" ;;
    chat2m-speech) echo "speech" ;;
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
    sherpa|sherpa-onnx)
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
        SenseVoiceSmall|sensevoice-small|sensevoicesmall)
          VOICE_ASR_MODEL="SenseVoiceSmall"
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
    *)
      known_value_error "VOICE_ASR_ENGINE" "$VOICE_ASR_ENGINE" "sherpa, sensevoice"
      ;;
  esac
}

resolve_tts_model() {
  VOICE_TTS_ENGINE="$(normalize_key "${VOICE_TTS_ENGINE:-cosyvoice}")"
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
    cosyvoice)
      VOICE_TTS_MODEL="${VOICE_TTS_MODEL:-CosyVoice-300M-SFT}"
      case "$VOICE_TTS_MODEL" in
        CosyVoice-300M-SFT|cosyvoice-300m-sft)
          VOICE_TTS_MODEL="CosyVoice-300M-SFT"
          TTS_HF_REPO_ID="FunAudioLLM/CosyVoice-300M-SFT"
          TTS_HF_REVISION="main"
          TTS_REQUIRED_FILES="cosyvoice.yaml,flow.pt,hift.pt,llm.pt,spk2info.pt,campplus.onnx,speech_tokenizer_v1.onnx"
          ;;
        CosyVoice-300M-Instruct|cosyvoice-300m-instruct)
          VOICE_TTS_MODEL="CosyVoice-300M-Instruct"
          TTS_HF_REPO_ID="FunAudioLLM/CosyVoice-300M-Instruct"
          TTS_HF_REVISION="main"
          TTS_REQUIRED_FILES="cosyvoice.yaml,flow.pt,hift.pt,llm.pt,spk2info.pt,campplus.onnx,speech_tokenizer_v1.onnx"
          ;;
        *)
          known_value_error "VOICE_TTS_MODEL" "$VOICE_TTS_MODEL" "CosyVoice-300M-SFT, CosyVoice-300M-Instruct"
          ;;
      esac
      ;;
    *)
      known_value_error "VOICE_TTS_ENGINE" "$VOICE_TTS_ENGINE" "piper, cosyvoice"
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

sync_runtime_env_defaults() {
  default_runtime="$DEFAULT_CONFIG_DIR/runtime.env"
  target_runtime="$RUNTIME_CONFIG_PATH"
  [ -f "$default_runtime" ] || return 0
  [ -f "$target_runtime" ] || return 0
  [ -w "$target_runtime" ] || return 0

  sync_lock_dir="$CONFIG_DIR/.runtime-env-sync.lock"
  sync_lock_waited=0
  while ! mkdir "$sync_lock_dir" 2>/dev/null; do
    sleep 1
    sync_lock_waited=$((sync_lock_waited + 1))
    if [ "$sync_lock_waited" -ge 60 ]; then
      echo "Removing stale runtime config sync lock: $sync_lock_dir" >&2
      rmdir "$sync_lock_dir" 2>/dev/null || true
      sync_lock_waited=0
    fi
  done
  trap 'rmdir "$sync_lock_dir" 2>/dev/null || true' EXIT

  appended=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    case "$key" in
      ''|*[!A-Za-z0-9_]*) continue ;;
    esac

    if ! grep -Eq "^[[:space:]]*$key[[:space:]]*=" "$target_runtime"; then
      if [ "$appended" -eq 0 ]; then
        printf '\n# Added by Chat2M image defaults. Existing values are never overwritten.\n' >> "$target_runtime"
        appended=1
      fi
      printf '%s\n' "$line" >> "$target_runtime"
      echo "Added missing runtime config: $key"
    fi
  done < "$default_runtime"

  rmdir "$sync_lock_dir"
  trap - EXIT
}

required_files_ok() {
  for required_file in "$@"; do
    if [ ! -s "$required_file" ]; then
      echo "Missing or empty model file: $required_file"
      return 1
    fi
  done
}

json_file_ok() {
  python3 - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        json.load(handle)
except Exception as exc:
    print(f"Invalid JSON file: {sys.argv[1]}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
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

piper_runtime_ok() {
  python3 - "$PIPER_DIR/model.onnx" "$PIPER_DIR/model.onnx.json" <<'PY'
import sys
from piper.voice import PiperVoice

try:
    PiperVoice.load(sys.argv[1], config_path=sys.argv[2])
except Exception as exc:
    print(f"Invalid Piper model: {sys.argv[1]}: {exc}", file=sys.stderr)
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

piper_model_ok() {
  required_files_ok \
    "$PIPER_DIR/model.onnx" \
    "$PIPER_DIR/model.onnx.json" \
    && json_file_ok "$PIPER_DIR/model.onnx.json" \
    && piper_runtime_ok
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

cosyvoice_model_ok() {
  required_relative_files_ok "$TTS_MODEL_DIR" "$TTS_REQUIRED_FILES"
}

cosyvoice_fp16_jit_ok() {
  required_files_ok \
    "$TTS_MODEL_DIR/llm.text_encoder.fp16.zip" \
    "$TTS_MODEL_DIR/flow.encoder.fp16.zip"
}

cosyvoice_fp16_trt_ok() {
  required_files_ok "$TTS_MODEL_DIR/flow.decoder.estimator.fp16.mygpu.plan"
}

ensure_cosyvoice_fp16_jit() {
  env_flag_enabled "${COSYVOICE_LOAD_JIT:-0}" || return 0

  if ! env_flag_enabled "${COSYVOICE_FP16:-0}"; then
    echo "COSYVOICE_LOAD_JIT=1 requires COSYVOICE_FP16=1; fp32 JIT is not supported in this image" >&2
    exit 1
  fi

  if cosyvoice_fp16_jit_ok; then
    echo "cosyvoice fp16 JIT artifacts are ready"
    return
  fi

  echo "[runtime] generating CosyVoice fp16 JIT artifacts"
  python3 - "$TTS_MODEL_DIR" <<'PY'
import importlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path


def install_torchaudio_stub():
    if "torchaudio" in sys.modules:
        return

    torchaudio = types.ModuleType("torchaudio")
    compliance = types.ModuleType("torchaudio.compliance")
    kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    transforms = types.ModuleType("torchaudio.transforms")
    torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    compliance.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance", loader=None)
    kaldi.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance.kaldi", loader=None)
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchaudio.transforms", loader=None)

    def unavailable(*args, **kwargs):
        raise RuntimeError("torchaudio is not available in CosyVoice inference runtime")

    class Resample:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, speech):
            return speech

    kaldi.fbank = unavailable
    torchaudio.load = unavailable
    torchaudio.save = unavailable
    transforms.Resample = Resample
    transforms.Spectrogram = unavailable
    transforms.MelSpectrogram = unavailable
    torchaudio.compliance = compliance
    torchaudio.transforms = transforms
    compliance.kaldi = kaldi
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.compliance.kaldi"] = kaldi
    sys.modules["torchaudio.transforms"] = transforms


def install_cosyvoice_stubs():
    install_torchaudio_stub()

    try:
        dataset_package = importlib.import_module("cosyvoice.dataset")
    except Exception:
        dataset_package = None

    if dataset_package is not None:
        processor = types.ModuleType("cosyvoice.dataset.processor")

        def unused_processor(*args, **kwargs):
            raise RuntimeError("CosyVoice dataset processor is not available in inference runtime")

        for name in (
            "parquet_opener",
            "tokenize",
            "filter",
            "resample",
            "compute_fbank",
            "parse_embedding",
            "shuffle",
            "sort",
            "batch",
            "padding",
        ):
            setattr(processor, name, unused_processor)
        sys.modules["cosyvoice.dataset.processor"] = processor
        setattr(dataset_package, "processor", processor)

    pylogger = types.ModuleType("matcha.utils.pylogger")

    def get_pylogger(name=__name__):
        import logging

        return logging.getLogger(name)

    pylogger.get_pylogger = get_pylogger
    sys.modules["matcha.utils.pylogger"] = pylogger

    def noop(*args, **kwargs):
        return None

    instantiators = types.ModuleType("matcha.utils.instantiators")
    instantiators.instantiate_callbacks = lambda *args, **kwargs: []
    instantiators.instantiate_loggers = lambda *args, **kwargs: []

    logging_utils = types.ModuleType("matcha.utils.logging_utils")
    logging_utils.log_hyperparameters = noop

    rich_utils = types.ModuleType("matcha.utils.rich_utils")
    rich_utils.enforce_tags = noop
    rich_utils.print_config_tree = noop

    utils = types.ModuleType("matcha.utils.utils")
    utils.extras = noop
    utils.get_metric_value = lambda *args, **kwargs: 0.0
    utils.task_wrapper = lambda func: func

    sys.modules["matcha.utils.instantiators"] = instantiators
    sys.modules["matcha.utils.logging_utils"] = logging_utils
    sys.modules["matcha.utils.rich_utils"] = rich_utils
    sys.modules["matcha.utils.utils"] = utils


def patch_onnxruntime_provider():
    import onnxruntime

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return

    original_inference_session = onnxruntime.InferenceSession

    def inference_session(*args, **kwargs):
        providers = kwargs.get("providers")
        if providers and "CUDAExecutionProvider" in providers:
            kwargs["providers"] = [provider for provider in providers if provider != "CUDAExecutionProvider"]
            if "CPUExecutionProvider" not in kwargs["providers"]:
                kwargs["providers"].append("CPUExecutionProvider")
        return original_inference_session(*args, **kwargs)

    onnxruntime.InferenceSession = inference_session


def save_fp16_script(module, target):
    import torch

    target = Path(target)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    module = module.half().eval()
    script = torch.jit.script(module)
    script = torch.jit.freeze(script)
    script = torch.jit.optimize_for_inference(script)
    script.save(str(tmp))
    os.replace(tmp, target)


def main():
    model_dir = Path(sys.argv[1])
    for path in reversed([item for item in os.environ.get("COSYVOICE_PACKAGE_PATH", "").split(":") if item]):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CosyVoice fp16 JIT export requires CUDA")

    install_cosyvoice_stubs()
    patch_onnxruntime_provider()

    from cosyvoice.cli.cosyvoice import CosyVoice

    torch._C._jit_set_fusion_strategy([("STATIC", 1)])
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    model = CosyVoice(str(model_dir), load_jit=False, load_trt=False, fp16=True)
    save_fp16_script(model.model.llm.text_encoder, model_dir / "llm.text_encoder.fp16.zip")
    save_fp16_script(model.model.flow.encoder, model_dir / "flow.encoder.fp16.zip")


if __name__ == "__main__":
    main()
PY

  if ! cosyvoice_fp16_jit_ok; then
    echo "CosyVoice fp16 JIT artifacts are still invalid after export" >&2
    exit 1
  fi
}

ensure_cosyvoice_fp16_trt() {
  env_flag_enabled "${COSYVOICE_LOAD_TRT:-0}" || return 0

  if ! env_flag_enabled "${COSYVOICE_FP16:-0}"; then
    echo "COSYVOICE_LOAD_TRT=1 requires COSYVOICE_FP16=1; fp32 TRT is not supported in this image" >&2
    exit 1
  fi

  ensure_cosyvoice_tensorrt_runtime

  if cosyvoice_fp16_trt_ok; then
    echo "cosyvoice fp16 TensorRT plan is ready"
    return
  fi

  echo "Missing CosyVoice fp16 TensorRT plan: $TTS_MODEL_DIR/flow.decoder.estimator.fp16.mygpu.plan" >&2
  echo "Disable COSYVOICE_LOAD_TRT or build the plan inside the speech container before startup." >&2
  exit 1
}

python_module_ok() {
  python3 - "$1" <<'PY'
import importlib.util
import os
import sys

for path in reversed([item for item in os.environ.get("COSYVOICE_PACKAGE_PATH", "").split(":") if item]):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    found = importlib.util.find_spec(sys.argv[1]) is not None
except ModuleNotFoundError:
    found = False
sys.exit(0 if found else 1)
PY
}

install_python_packages() {
  label="$1"
  shift

  echo "[runtime] installing $label dependencies"
  mkdir -p "$PYTHON_RUNTIME_PREFIX"
  python3 -m pip install --user --no-cache-dir --retries "${PIP_RETRIES:-10}" --timeout "${PIP_TIMEOUT:-300}" "$@"
}

install_python_package_url() {
  label="$1"
  url="$2"

  echo "[runtime] installing $label dependency"
  mkdir -p "$PYTHON_RUNTIME_PREFIX"
  python3 -m pip install --user --no-cache-dir --retries "${PIP_RETRIES:-10}" --timeout "${PIP_TIMEOUT:-300}" "$url"
}

install_python_package_url_force() {
  label="$1"
  url="$2"

  echo "[runtime] installing $label dependency"
  mkdir -p "$PYTHON_RUNTIME_PREFIX"
  python3 -m pip install --user --no-cache-dir --force-reinstall --no-deps --retries "${PIP_RETRIES:-10}" --timeout "${PIP_TIMEOUT:-300}" "$url"
}

ensure_apt_packages() {
  missing=""
  for package in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
      missing="$missing $package"
    fi
  done
  [ -n "$missing" ] || return 0

  echo "[runtime] installing system dependencies:$missing"
  apt-get -o Acquire::Retries="${APT_RETRIES:-5}" update
  apt-get -o Acquire::Retries="${APT_RETRIES:-5}" install -y --fix-missing --no-install-recommends $missing
  rm -rf /var/lib/apt/lists/*
}

ensure_jetson_apt_sources() {
  if [ ! -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list ]; then
    echo "[runtime] adding NVIDIA Jetson apt sources"
    curl -fsSL --retry 5 --connect-timeout 20 https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
      -o /etc/apt/trusted.gpg.d/jetson-ota-public.asc
    jetson_platform="${JETSON_L4T_PLATFORM:-t234}"
    cat > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list <<'EOF'
deb https://repo.download.nvidia.com/jetson/common r35.6 main
EOF
    printf 'deb https://repo.download.nvidia.com/jetson/%s r35.6 main\n' "$jetson_platform" \
      >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
  fi
}

ensure_jetson_cuda_runtime() {
  ensure_jetson_apt_sources
  ensure_apt_packages \
    cuda-cudart-11-4 \
    cuda-cupti-11-4 \
    cuda-nvrtc-11-4 \
    cuda-nvtx-11-4 \
    libcublas-11-4 \
    libcufft-11-4 \
    libcudnn8 \
    libcurand-11-4 \
    libcusolver-11-4 \
    libcusparse-11-4 \
    libnpp-11-4
}

ensure_cosyvoice_tensorrt_runtime() {
  ensure_jetson_apt_sources
  ensure_apt_packages \
    libcudla-11-4 \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvonnxparsers8 \
    libnvparsers8 \
    python3-libnvinfer \
    tensorrt-libs \
    libnvinfer-bin

  if ! python_module_ok onnx; then
    install_python_packages "ONNX runtime export helper" "onnx==1.16.1"
  fi
  python_module_ok tensorrt
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

link_jetson_cuda_python_libs() {
  nvidia_dir="$PYTHON_RUNTIME_SITE/nvidia"
  mkdir -p "$nvidia_dir"

  link_cuda_lib_dir cuda_runtime "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libcudart.so*"
  link_cuda_lib_dir cuda_cupti "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libcupti.so*"
  link_cuda_lib_dir cuda_nvrtc "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libnvrtc.so*"
  link_cuda_lib_dir cublas "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libcublas.so*" "libcublasLt.so*"
  link_cuda_lib_dir cudnn "/usr/lib/aarch64-linux-gnu" "libcudnn.so*"
  link_cuda_lib_dir cufft "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libcufft.so*"
  link_cuda_lib_dir curand "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libcurand.so*"
  link_cuda_lib_dir cusolver "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libcusolver.so*"
  link_cuda_lib_dir cusparse "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libcusparse.so*"
  link_cuda_lib_dir nccl "/usr/lib/aarch64-linux-gnu" "libnccl.so*"
  link_cuda_lib_dir nvtx "/usr/local/cuda-11.4/targets/aarch64-linux/lib" "libnvToolsExt.so*"
}

link_cuda_lib_dir() {
  package_name="$1"
  source_dir="$2"
  shift 2
  target_dir="$PYTHON_RUNTIME_SITE/nvidia/$package_name/lib"
  mkdir -p "$target_dir"

  for pattern in "$@"; do
    for source_file in "$source_dir"/$pattern; do
      [ -e "$source_file" ] || continue
      ln -sf "$source_file" "$target_dir/$(basename "$source_file")"
    done
  done
}

cosyvoice_cuda_requested() {
  [ "${VOICE_TTS_ENGINE:-}" = "cosyvoice" ] || return 1
  case "$(normalize_key "${VOICE_TTS_DEVICE:-cuda}")" in
    cuda|gpu|cuda:*) return 0 ;;
    auto) return 0 ;;
    *)
      echo "CosyVoice requires GPU. Set VOICE_TTS_DEVICE=cuda or auto." >&2
      exit 1
      ;;
  esac
}

install_cosyvoice_torch() {
  cosyvoice_cuda_requested
  ensure_jetson_cuda_runtime
  link_jetson_cuda_python_libs
  if verify_torch_cuda_runtime; then
    return
  fi
  install_python_package_url_force \
    "Jetson CUDA PyTorch" \
    "${JETSON_TORCH_WHEEL_URL:-https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl}"
  link_jetson_cuda_python_libs
  verify_torch_cuda_runtime
}

ensure_kws_runtime() {
  if python_module_ok sherpa_onnx && command -v sherpa-onnx-cli >/dev/null 2>&1; then
    return
  fi
  install_python_packages "sherpa-onnx" "sherpa-onnx==1.12.38"
}

ensure_sherpa_asr_runtime() {
  ensure_kws_runtime
}

ensure_piper_runtime() {
  if python_module_ok piper; then
    return
  fi
  install_python_packages "Piper" "piper-tts==1.2.0"
}

sensevoice_runtime_ok() {
  python_module_ok sense_voice_streaming_asr \
    && python_module_ok onnxruntime \
    && python_module_ok kaldi_native_fbank \
    && python_module_ok sentencepiece
}

patch_sensevoice_runtime() {
  python3 <<'PY'
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
}

ensure_sensevoice_runtime() {
  if sensevoice_runtime_ok; then
    patch_sensevoice_runtime
    return
  fi
  install_python_packages "SenseVoice streaming ASR" "sense-voice-streaming-asr==0.1.1" "onnxruntime==${ONNXRUNTIME_VERSION:-1.16.3}" "sentencepiece"
  patch_sensevoice_runtime
}

cosyvoice_runtime_ok() {
  python_module_ok torch \
    && python_module_ok onnxruntime \
    && python_module_ok hyperpyyaml \
    && python_module_ok transformers \
    && python_module_ok whisper \
    && python_module_ok conformer \
    && python_module_ok diffusers \
    && python_module_ok einops \
    && python_module_ok librosa \
    && python_module_ok tiktoken \
    && python_module_ok inflect \
    && python_module_ok omegaconf \
    && python_module_ok scipy \
    && python_module_ok regex \
    && python_module_ok modelscope \
    && python_module_ok cosyvoice.cli.cosyvoice \
    && python_module_ok matcha
}

ensure_cosyvoice_runtime() {
  COSYVOICE_CODE_DIR="${COSYVOICE_CODE_DIR:-$MODELS_DIR/runtime/CosyVoice}"
  export COSYVOICE_PACKAGE_PATH="$COSYVOICE_CODE_DIR:$COSYVOICE_CODE_DIR/third_party/Matcha-TTS"
  if [ -d "$COSYVOICE_CODE_DIR/cosyvoice" ] \
    && [ -d "$COSYVOICE_CODE_DIR/third_party/Matcha-TTS/matcha" ] \
    && cosyvoice_runtime_ok; then
    cosyvoice_cuda_requested
    ensure_jetson_cuda_runtime
    if env_flag_enabled "${COSYVOICE_LOAD_TRT:-0}"; then
      ensure_cosyvoice_tensorrt_runtime
    fi
    verify_torch_cuda_runtime
    return
  fi

  ensure_apt_packages git
  install_cosyvoice_torch
  install_python_packages \
    "CosyVoice runtime" \
    "onnxruntime==${ONNXRUNTIME_VERSION:-1.16.3}" \
    "accelerate==0.34.2" \
    "conformer==0.3.2" \
    "diffusers==0.29.0" \
    "einops==0.8.0" \
    "hydra-core" \
    "HyperPyYAML==1.2.3" \
    "inflect==7.3.1" \
    "librosa==0.10.2" \
    "modelscope==1.20.0" \
    "omegaconf==2.3.0" \
    "openai-whisper==20231117" \
    "regex" \
    "scipy==1.10.1" \
    "sentencepiece" \
    "tiktoken==0.7.0" \
    "transformers==4.45.2"
  if [ ! -d "$COSYVOICE_CODE_DIR/cosyvoice" ]; then
    echo "[runtime] downloading CosyVoice code"
    rm -rf "$COSYVOICE_CODE_DIR"
    mkdir -p "$(dirname "$COSYVOICE_CODE_DIR")"
    git clone --depth 1 --branch "${COSYVOICE_GIT_REF:-v2.0}" https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_CODE_DIR"
  fi
  if [ ! -d "$COSYVOICE_CODE_DIR/third_party/Matcha-TTS/matcha" ]; then
    echo "[runtime] downloading Matcha-TTS code"
    rm -rf "$COSYVOICE_CODE_DIR/third_party/Matcha-TTS"
    mkdir -p "$COSYVOICE_CODE_DIR/third_party"
    git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git "$COSYVOICE_CODE_DIR/third_party/Matcha-TTS"
  fi
}

ensure_selected_runtimes() {
  if model_selected kws; then
    ensure_kws_runtime
  fi

  if model_selected speech; then
    case "$VOICE_ASR_ENGINE" in
      sherpa) ensure_sherpa_asr_runtime ;;
      sensevoice) ensure_sensevoice_runtime ;;
    esac
    case "$VOICE_TTS_ENGINE" in
      piper) ensure_piper_runtime ;;
      cosyvoice) ensure_cosyvoice_runtime ;;
    esac
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

  mkdir -p "$(dirname "$output")"
  total="$(content_length "$url" || true)"
  echo "[models] downloading $label"
  if [ -f "$tmp" ]; then
    completed="$(wc -c < "$tmp" | tr -d ' ')"
    echo "[models] resuming $label from $(awk -v done="$completed" 'BEGIN { printf("%.1f", done / 1048576) }') MB"
  else
    completed=0
  fi
  print_download_progress "$label" "$completed" "$total"

  curl -fL --retry 10 --retry-all-errors --connect-timeout 20 --speed-limit 1024 --speed-time 120 --continue-at - --silent --show-error "$url" -o "$tmp" &
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

  if ! wait "$curl_pid"; then
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
    rm -f "$tmp"
    return 1
  fi

  kill "$progress_pid" 2>/dev/null || true
  wait "$progress_pid" 2>/dev/null || true
  completed="$(wc -c < "$tmp" | tr -d ' ')"
  print_download_progress "$label" "$completed" "$total"
  mv "$tmp" "$output"
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

download_file() {
  output="$1"
  url="$2"

  mkdir -p "$(dirname "$output")"
  rm -f "$output"
  download_with_progress "$output" "$url" "$(basename "$output")"
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

ensure_piper_model() {
  if piper_model_ok; then
    echo "piper $VOICE_TTS_MODEL is ready"
    return
  fi

  echo "piper $VOICE_TTS_MODEL is missing or invalid; re-downloading"
  rm -rf "$PIPER_DIR"
  download_file "$PIPER_DIR/model.onnx" "$PIPER_MODEL_URL"
  download_file "$PIPER_DIR/model.onnx.json" "$PIPER_CONFIG_URL"

  echo "[models] validating piper $VOICE_TTS_MODEL"
  if ! piper_model_ok; then
    echo "piper $VOICE_TTS_MODEL is still invalid after download" >&2
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

init_config
sync_runtime_env_defaults
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
PIPER_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
TTS_MODEL_DIR="$MODELS_DIR/$VOICE_TTS_ENGINE/$VOICE_TTS_MODEL"
export VOICE_ASR_ENGINE
export VOICE_ASR_MODEL
export VOICE_TTS_ENGINE
export VOICE_TTS_MODEL
export SENSEVOICE_MODEL_DIR="$ASR_MODEL"
export SENSEVOICE_VAD_MODEL_DIR="$VAD_MODEL_DIR"
export COSYVOICE_PACKAGE_PATH="${COSYVOICE_PACKAGE_PATH:-}"
MODEL_SET="$(default_voice_model_set)"
if model_selected speech; then
  case "$VOICE_ASR_ENGINE" in
    sherpa) MODEL_SET="$MODEL_SET,asr" ;;
    sensevoice) MODEL_SET="$MODEL_SET,sensevoice" ;;
  esac
  case "$VOICE_TTS_ENGINE" in
    piper) MODEL_SET="$MODEL_SET,piper" ;;
    cosyvoice) MODEL_SET="$MODEL_SET,cosyvoice" ;;
  esac
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
  ensure_archive_model \
    "$VOICE_ASR_MODEL" \
    "$ASR_MODEL_URL" \
    asr_model_ok \
    "$ASR_MODEL"
fi

if model_selected piper; then
  ensure_piper_model
fi

if model_selected sensevoice; then
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

if model_selected cosyvoice; then
  ensure_hf_snapshot_model \
    "$VOICE_TTS_MODEL" \
    "$TTS_HF_REPO_ID" \
    "$TTS_HF_REVISION" \
    cosyvoice_model_ok \
    "$TTS_MODEL_DIR" \
    "$TTS_REQUIRED_FILES"
  ensure_cosyvoice_fp16_jit
  ensure_cosyvoice_fp16_trt
fi

trap - EXIT
rmdir "$LOCK_DIR"

exec "$@"
