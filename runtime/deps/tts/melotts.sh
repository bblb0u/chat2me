#!/bin/sh
set -eu

. /opt/chat2me-deps/lib/common.sh

/opt/chat2me-deps/platform/jetson_gpu.sh
/opt/chat2me-deps/platform/jetson_torch.sh

MELOTTS_SOURCE_REF="${MELOTTS_SOURCE_REF:-209145371cff8fc3bd60d7be902ea69cbdb7965a}"
MELOTTS_SOURCE_URL="${MELOTTS_SOURCE_URL:-https://github.com/myshell-ai/MeloTTS/archive/${MELOTTS_SOURCE_REF}.tar.gz}"

apt_install_packages \
  libsndfile1
rm -rf /var/lib/apt/lists/*

pip_install \
  "anyascii==0.3.2" \
  "cached_path" \
  "cn2an==0.5.22" \
  "eng_to_ipa==0.0.2" \
  "g2p_en==2.1.0" \
  "huggingface-hub==0.25.2" \
  "inflect==7.0.0" \
  "jieba==0.42.1" \
  "langid==1.1.6" \
  "loguru==0.7.2" \
  "num2words==0.5.12" \
  "pydub==0.25.1" \
  "pypinyin==0.50.0" \
  "scipy==1.10.1" \
  "soundfile==0.12.1" \
  "tqdm==4.66.4" \
  "transformers==4.27.4" \
    "unidecode==1.3.7"

rm -rf /opt/MeloTTS /tmp/melotts-source
mkdir -p /tmp/melotts-source
download_file "$MELOTTS_SOURCE_URL" /tmp/melotts-source/melotts.tar.gz "MeloTTS source"
tar -xzf /tmp/melotts-source/melotts.tar.gz -C /tmp/melotts-source
source_dir="$(find /tmp/melotts-source -mindepth 1 -maxdepth 1 -type d -name 'MeloTTS-*' | head -n 1)"
if [ -z "$source_dir" ]; then
  echo "MeloTTS source archive did not contain a MeloTTS-* directory" >&2
  exit 1
fi
mv "$source_dir" /opt/MeloTTS
rm -rf /tmp/melotts-source

python3 <<'PY'
from pathlib import Path

root = Path("/opt/MeloTTS")

cleaner = root / "melo" / "text" / "cleaner.py"
cleaner_text = cleaner.read_text(encoding="utf-8")
cleaner_text = cleaner_text.replace(
    "from . import chinese, japanese, english, chinese_mix, korean, french, spanish\n",
    "from . import chinese, english, chinese_mix\n",
)
cleaner_text = cleaner_text.replace(
    'language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, \'ZH_MIX_EN\': chinese_mix, \'KR\': korean,\n'
    "                    'FR': french, 'SP': spanish, 'ES': spanish}\n",
    'language_module_map = {"ZH": chinese, "EN": english, "ZH_MIX_EN": chinese_mix}\n',
)
cleaner.write_text(cleaner_text, encoding="utf-8")

english = root / "melo" / "text" / "english.py"
english_text = english.read_text(encoding="utf-8")
english_text = english_text.replace(
    "from .japanese import distribute_phone\n",
    "\n"
    "def distribute_phone(n_phone, n_word):\n"
    "    phones_per_word = [0] * n_word\n"
    "    for _ in range(n_phone):\n"
    "        min_tasks = min(phones_per_word)\n"
    "        min_index = phones_per_word.index(min_tasks)\n"
    "        phones_per_word[min_index] += 1\n"
    "    return phones_per_word\n",
)
english.write_text(english_text, encoding="utf-8")

torchaudio = root / "torchaudio"
torchaudio.mkdir(exist_ok=True)
(torchaudio / "__init__.py").write_text(
    "import numpy as np\n"
    "import soundfile as sf\n"
    "import torch\n\n"
    "def load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True):\n"
    "    data, sample_rate = sf.read(path, dtype='float32' if normalize else 'int16', always_2d=True)\n"
    "    if frame_offset:\n"
    "        data = data[frame_offset:]\n"
    "    if num_frames is not None and num_frames >= 0:\n"
    "        data = data[:num_frames]\n"
    "    if not normalize and data.dtype != np.float32:\n"
    "        data = data.astype(np.float32) / 32768.0\n"
    "    tensor = torch.from_numpy(data.T.copy() if channels_first else data.copy())\n"
    "    return tensor, int(sample_rate)\n",
    encoding="utf-8",
)

librosa = root / "librosa"
librosa.mkdir(exist_ok=True)
(librosa / "__init__.py").write_text(
    "import numpy as np\n"
    "import soundfile as sf\n\n"
    "def load(path, sr=22050, mono=True):\n"
    "    data, sample_rate = sf.read(path, dtype='float32', always_2d=True)\n"
    "    if mono:\n"
    "        data = data.mean(axis=1)\n"
    "    else:\n"
    "        data = data.T\n"
    "    if sr is not None and int(sample_rate) != int(sr):\n"
    "        raise RuntimeError('librosa stub cannot resample audio')\n"
    "    return np.asarray(data, dtype=np.float32), int(sample_rate)\n",
    encoding="utf-8",
)
PY
