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
  "cached_path==1.6.3" \
  "cn2an==0.5.22" \
  "eng_to_ipa==0.0.2" \
  "g2p_en==2.1.0" \
  "huggingface-hub==0.23.5" \
  "inflect==7.0.0" \
  "jieba==0.42.1" \
  "langid==1.1.6" \
  "loguru==0.7.2" \
  "num2words==0.5.12" \
  "numba==0.58.1" \
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
    "from importlib import import_module\n"
    "from . import chinese\n",
)
cleaner_text = cleaner_text.replace(
    'language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, \'ZH_MIX_EN\': chinese_mix, \'KR\': korean,\n'
    "                    'FR': french, 'SP': spanish, 'ES': spanish}\n",
    'language_module_map = {"ZH": chinese}\n\n'
    "def get_language_module(language):\n"
    "    if language in language_module_map:\n"
    "        return language_module_map[language]\n"
    '    module_name_map = {"EN": "english", "ZH_MIX_EN": "chinese_mix"}\n'
    "    module_name = module_name_map.get(language)\n"
    "    if module_name is None:\n"
    "        raise KeyError(language)\n"
    '    module = import_module("." + module_name, __package__)\n'
    "    language_module_map[language] = module\n"
    "    return module\n",
)
cleaner_text = cleaner_text.replace(
    "language_module = language_module_map[language]",
    "language_module = get_language_module(language)",
)
cleaner.write_text(cleaner_text, encoding="utf-8")

text_init = root / "melo" / "text" / "__init__.py"
text_init_text = text_init.read_text(encoding="utf-8")
text_init_text = text_init_text.replace(
    "def get_bert(norm_text, word2ph, language, device):\n"
    "    from .chinese_bert import get_bert_feature as zh_bert\n"
    "    from .english_bert import get_bert_feature as en_bert\n"
    "    from .japanese_bert import get_bert_feature as jp_bert\n"
    "    from .chinese_mix import get_bert_feature as zh_mix_en_bert\n"
    "    from .spanish_bert import get_bert_feature as sp_bert\n"
    "    from .french_bert import get_bert_feature as fr_bert\n"
    "    from .korean import get_bert_feature as kr_bert\n"
    "\n"
    "    lang_bert_func_map = {\"ZH\": zh_bert, \"EN\": en_bert, \"JP\": jp_bert, 'ZH_MIX_EN': zh_mix_en_bert, \n"
    "                          'FR': fr_bert, 'SP': sp_bert, 'ES': sp_bert, \"KR\": kr_bert}\n"
    "    bert = lang_bert_func_map[language](norm_text, word2ph, device)\n"
    "    return bert\n",
    "def get_bert(norm_text, word2ph, language, device):\n"
    "    if language == \"ZH\":\n"
    "        from .chinese_bert import get_bert_feature\n"
    "    elif language == \"EN\":\n"
    "        from .english_bert import get_bert_feature\n"
    "    elif language == \"ZH_MIX_EN\":\n"
    "        from .chinese_mix import get_bert_feature\n"
    "    else:\n"
    "        raise KeyError(language)\n"
    "    return get_bert_feature(norm_text, word2ph, device)\n",
)
text_init.write_text(text_init_text, encoding="utf-8")

api = root / "melo" / "api.py"
api_text = api.read_text(encoding="utf-8")
api_text = api_text.replace(
    "        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model\n",
    "        self.language = language\n",
)
api.write_text(api_text, encoding="utf-8")

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

g2p_en = Path("/usr/local/lib/python3.8/dist-packages/g2p_en/g2p.py")
if g2p_en.is_file():
    g2p_en_text = g2p_en.read_text(encoding="utf-8")
    g2p_en_text = g2p_en_text.replace(
        "try:\n"
        "    nltk.data.find('taggers/averaged_perceptron_tagger.zip')\n"
        "except LookupError:\n"
        "    nltk.download('averaged_perceptron_tagger')\n"
        "try:\n"
        "    nltk.data.find('corpora/cmudict.zip')\n"
        "except LookupError:\n"
        "    nltk.download('cmudict')\n",
        "",
    )
    g2p_en.write_text(g2p_en_text, encoding="utf-8")

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
    "from . import filters, util\n\n"
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
(librosa / "filters.py").write_text(
    "import numpy as np\n\n"
    "def _hz_to_mel(frequencies):\n"
    "    frequencies = np.asanyarray(frequencies)\n"
    "    return 2595.0 * np.log10(1.0 + frequencies / 700.0)\n\n"
    "def _mel_to_hz(mels):\n"
    "    mels = np.asanyarray(mels)\n"
    "    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)\n\n"
    "def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, dtype=np.float32, norm='slaney', htk=False):\n"
    "    if fmax is None:\n"
    "        fmax = float(sr) / 2.0\n"
    "    fftfreqs = np.linspace(0.0, float(sr) / 2.0, int(1 + n_fft // 2))\n"
    "    min_mel = _hz_to_mel(fmin)\n"
    "    max_mel = _hz_to_mel(fmax)\n"
    "    mel_f = _mel_to_hz(np.linspace(min_mel, max_mel, int(n_mels) + 2))\n"
    "    fdiff = np.diff(mel_f)\n"
    "    ramps = np.subtract.outer(mel_f, fftfreqs)\n"
    "    weights = np.zeros((int(n_mels), int(1 + n_fft // 2)), dtype=dtype)\n"
    "    for i in range(int(n_mels)):\n"
    "        lower = -ramps[i] / fdiff[i]\n"
    "        upper = ramps[i + 2] / fdiff[i + 1]\n"
    "        weights[i] = np.maximum(0.0, np.minimum(lower, upper))\n"
    "    if norm == 'slaney':\n"
    "        enorm = 2.0 / (mel_f[2:int(n_mels) + 2] - mel_f[:int(n_mels)])\n"
    "        weights *= enorm[:, np.newaxis]\n"
    "    return weights.astype(dtype, copy=False)\n",
    encoding="utf-8",
)
(librosa / "util.py").write_text(
    "import numpy as np\n\n"
    "def pad_center(data, *, size, axis=-1, **kwargs):\n"
    "    n = data.shape[axis]\n"
    "    if n > size:\n"
    "        raise ValueError('target size must be at least input size')\n"
    "    lpad = int((size - n) // 2)\n"
    "    rpad = int(size - n - lpad)\n"
    "    try:\n"
    "        import torch\n"
    "    except Exception:\n"
    "        torch = None\n"
    "    if torch is not None and isinstance(data, torch.Tensor):\n"
    "        axis = axis if axis >= 0 else data.dim() + axis\n"
    "        if axis != data.dim() - 1:\n"
    "            data = data.transpose(axis, -1)\n"
    "            out = torch.nn.functional.pad(data, (lpad, rpad))\n"
    "            return out.transpose(axis, -1)\n"
    "        return torch.nn.functional.pad(data, (lpad, rpad))\n"
    "    pad_width = [(0, 0)] * np.ndim(data)\n"
    "    pad_width[axis] = (lpad, rpad)\n"
    "    return np.pad(data, pad_width, **kwargs)\n",
    encoding="utf-8",
)
PY
