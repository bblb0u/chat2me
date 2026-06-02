from __future__ import annotations

import base64
import string
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")


def install_cosyvoice_runtime_adapters(package_path: str, whisper_assets_dir: Path, text_frontend: bool) -> None:
    import importlib
    import importlib.machinery
    import types

    for path in reversed([item.strip() for item in package_path.split(":") if item.strip()]):
        if path not in sys.path:
            sys.path.insert(0, path)

    if importlib.util.find_spec("torchaudio") is None:
        from app.f5_tts_runtime import install_torchaudio_shim

        install_torchaudio_shim()

    if importlib.util.find_spec("whisper") is None:
        whisper = types.ModuleType("whisper")
        whisper.__spec__ = importlib.machinery.ModuleSpec("whisper", loader=None)
        tokenizer = types.ModuleType("whisper.tokenizer")
        tokenizer.__spec__ = importlib.machinery.ModuleSpec("whisper.tokenizer", loader=None)

        languages = {
            "en": "english",
            "zh": "chinese",
            "de": "german",
            "es": "spanish",
            "ru": "russian",
            "ko": "korean",
            "fr": "french",
            "ja": "japanese",
            "pt": "portuguese",
            "tr": "turkish",
            "pl": "polish",
            "ca": "catalan",
            "nl": "dutch",
            "ar": "arabic",
            "sv": "swedish",
            "it": "italian",
            "id": "indonesian",
            "hi": "hindi",
            "fi": "finnish",
            "vi": "vietnamese",
            "he": "hebrew",
            "uk": "ukrainian",
            "el": "greek",
            "ms": "malay",
            "cs": "czech",
            "ro": "romanian",
            "da": "danish",
            "hu": "hungarian",
            "ta": "tamil",
            "no": "norwegian",
            "th": "thai",
            "ur": "urdu",
            "hr": "croatian",
            "bg": "bulgarian",
            "lt": "lithuanian",
            "la": "latin",
            "mi": "maori",
            "ml": "malayalam",
            "cy": "welsh",
            "sk": "slovak",
            "te": "telugu",
            "fa": "persian",
            "lv": "latvian",
            "bn": "bengali",
            "sr": "serbian",
            "az": "azerbaijani",
            "sl": "slovenian",
            "kn": "kannada",
            "et": "estonian",
            "mk": "macedonian",
            "br": "breton",
            "eu": "basque",
            "is": "icelandic",
            "hy": "armenian",
            "ne": "nepali",
            "mn": "mongolian",
            "bs": "bosnian",
            "kk": "kazakh",
            "sq": "albanian",
            "sw": "swahili",
            "gl": "galician",
            "mr": "marathi",
            "pa": "punjabi",
            "si": "sinhala",
            "km": "khmer",
            "sn": "shona",
            "yo": "yoruba",
            "so": "somali",
            "af": "afrikaans",
            "oc": "occitan",
            "ka": "georgian",
            "be": "belarusian",
            "tg": "tajik",
            "sd": "sindhi",
            "gu": "gujarati",
            "am": "amharic",
            "yi": "yiddish",
            "lo": "lao",
            "uz": "uzbek",
            "fo": "faroese",
            "ht": "haitian creole",
            "ps": "pashto",
            "tk": "turkmen",
            "nn": "nynorsk",
            "mt": "maltese",
            "sa": "sanskrit",
            "lb": "luxembourgish",
            "my": "myanmar",
            "bo": "tibetan",
            "tl": "tagalog",
            "mg": "malagasy",
            "as": "assamese",
            "tt": "tatar",
            "haw": "hawaiian",
            "ln": "lingala",
            "ha": "hausa",
            "ba": "bashkir",
            "jw": "javanese",
            "su": "sundanese",
            "yue": "cantonese",
        }
        language_aliases = {
            **{language: code for code, language in languages.items()},
            "burmese": "my",
            "valencian": "ca",
            "flemish": "nl",
            "haitian": "ht",
            "letzeburgesch": "lb",
            "pushto": "ps",
            "panjabi": "pa",
            "moldavian": "ro",
            "moldovan": "ro",
            "sinhalese": "si",
            "castilian": "es",
            "mandarin": "zh",
        }

        @dataclass
        class Tokenizer:
            encoding: Any
            num_languages: int
            language: str | None = None
            task: str | None = None
            sot_sequence: tuple[int, ...] = ()
            special_tokens: dict[str, int] = field(default_factory=dict)

            def __post_init__(self) -> None:
                for special in self.encoding.special_tokens_set:
                    self.special_tokens[special] = self.encoding.encode_single_token(special)
                sot = self.special_tokens["<|startoftranscript|>"]
                translate = self.special_tokens["<|translate|>"]
                transcribe = self.special_tokens["<|transcribe|>"]
                language_codes = tuple(languages.keys())[: self.num_languages]
                sequence = [sot]
                if self.language is not None:
                    sequence.append(sot + 1 + language_codes.index(self.language))
                if self.task is not None:
                    sequence.append(transcribe if self.task == "transcribe" else translate)
                self.sot_sequence = tuple(sequence)

            def encode(self, text: str, **kwargs: Any) -> list[int]:
                return list(self.encoding.encode(text, **kwargs))

            def decode(self, token_ids: list[int], **kwargs: Any) -> str:
                return str(self.encoding.decode([token for token in token_ids if token < self.timestamp_begin], **kwargs))

            def decode_with_timestamps(self, token_ids: list[int], **kwargs: Any) -> str:
                return str(self.encoding.decode(token_ids, **kwargs))

            @cached_property
            def eot(self) -> int:
                return int(self.encoding.eot_token)

            @cached_property
            def transcribe(self) -> int:
                return self.special_tokens["<|transcribe|>"]

            @cached_property
            def translate(self) -> int:
                return self.special_tokens["<|translate|>"]

            @cached_property
            def sot(self) -> int:
                return self.special_tokens["<|startoftranscript|>"]

            @cached_property
            def sot_lm(self) -> int:
                return self.special_tokens["<|startoflm|>"]

            @cached_property
            def sot_prev(self) -> int:
                return self.special_tokens["<|startofprev|>"]

            @cached_property
            def no_speech(self) -> int:
                return self.special_tokens["<|nospeech|>"]

            @cached_property
            def no_timestamps(self) -> int:
                return self.special_tokens["<|notimestamps|>"]

            @cached_property
            def timestamp_begin(self) -> int:
                return self.special_tokens["<|0.00|>"]

            @cached_property
            def language_token(self) -> int:
                if self.language is None:
                    raise ValueError("This tokenizer does not have language token configured")
                return self.to_language_token(self.language)

            def to_language_token(self, language: str) -> int:
                token = self.special_tokens.get(f"<|{language}|>")
                if token is None:
                    raise KeyError(f"Language {language} not found in tokenizer.")
                return token

            @cached_property
            def all_language_tokens(self) -> tuple[int, ...]:
                result = []
                for token, token_id in self.special_tokens.items():
                    if token.strip("<|>") in languages:
                        result.append(token_id)
                return tuple(result)[: self.num_languages]

            @cached_property
            def all_language_codes(self) -> tuple[str, ...]:
                return tuple(self.decode([token]).strip("<|>") for token in self.all_language_tokens)

            @cached_property
            def sot_sequence_including_notimestamps(self) -> tuple[int, ...]:
                return tuple(list(self.sot_sequence) + [self.no_timestamps])

            @cached_property
            def non_speech_tokens(self) -> tuple[int, ...]:
                symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
                symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }}".split()
                miscellaneous = set("♩♪♫♬♭♮♯")
                result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
                for symbol in symbols + list(miscellaneous):
                    for tokens in (self.encoding.encode(symbol), self.encoding.encode(" " + symbol)):
                        if len(tokens) == 1 or symbol in miscellaneous:
                            result.add(tokens[0])
                return tuple(sorted(result))

            def split_to_word_tokens(self, tokens: list[int]) -> Any:
                if self.language in {"zh", "ja", "th", "lo", "my", "yue"}:
                    return self.split_tokens_on_unicode(tokens)
                return self.split_tokens_on_spaces(tokens)

            def split_tokens_on_unicode(self, tokens: list[int]) -> Any:
                decoded_full = self.decode_with_timestamps(tokens)
                replacement_char = "\ufffd"
                words = []
                word_tokens = []
                current_tokens = []
                unicode_offset = 0
                for token in tokens:
                    current_tokens.append(token)
                    decoded = self.decode_with_timestamps(current_tokens)
                    if replacement_char not in decoded or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char:
                        words.append(decoded)
                        word_tokens.append(current_tokens)
                        current_tokens = []
                        unicode_offset += len(decoded)
                return words, word_tokens

            def split_tokens_on_spaces(self, tokens: list[int]) -> Any:
                subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
                words = []
                word_tokens = []
                for subword, subword_tokens in zip(subwords, subword_tokens_list):
                    special = subword_tokens[0] >= self.eot
                    with_space = subword.startswith(" ")
                    punctuation = subword.strip() in string.punctuation
                    if special or with_space or punctuation or len(words) == 0:
                        words.append(subword)
                        word_tokens.append(subword_tokens)
                    else:
                        words[-1] = words[-1] + subword
                        word_tokens[-1].extend(subword_tokens)
                return words, word_tokens

        def get_encoding(name: str = "gpt2", num_languages: int = 99) -> Any:
            import tiktoken

            vocab_path = whisper_assets_dir / f"{name}.tiktoken"
            require_file(vocab_path)
            ranks = {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line)
            }
            n_vocab = len(ranks)
            special_tokens: dict[str, int] = {}
            specials = [
                "<|endoftext|>",
                "<|startoftranscript|>",
                *[f"<|{language}|>" for language in list(languages.keys())[:num_languages]],
                "<|translate|>",
                "<|transcribe|>",
                "<|startoflm|>",
                "<|startofprev|>",
                "<|nospeech|>",
                "<|notimestamps|>",
                *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
            ]
            for special in specials:
                special_tokens[special] = n_vocab
                n_vocab += 1
            return tiktoken.Encoding(
                name=vocab_path.name,
                explicit_n_vocab=n_vocab,
                pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                mergeable_ranks=ranks,
                special_tokens=special_tokens,
            )

        def get_tokenizer(multilingual: bool, *args: Any, **kwargs: Any) -> Tokenizer:
            num_languages = int(kwargs.get("num_languages", 99))
            language = kwargs.get("language")
            task = kwargs.get("task")
            if language is not None:
                language = str(language).lower()
                if language not in languages:
                    if language in language_aliases:
                        language = language_aliases[language]
                    else:
                        raise ValueError(f"Unsupported language: {language}")
            if multilingual:
                encoding_name = "multilingual"
                language = language or "en"
                task = task or "transcribe"
            else:
                encoding_name = "gpt2"
                language = None
                task = None
            return Tokenizer(
                encoding=get_encoding(name=encoding_name, num_languages=num_languages),
                num_languages=num_languages,
                language=language,
                task=task,
            )

        def log_mel_spectrogram(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("whisper is not available in this fixed-speaker CosyVoice inference image")

        tokenizer.Tokenizer = Tokenizer
        tokenizer.get_encoding = get_encoding
        tokenizer.get_tokenizer = get_tokenizer
        tokenizer.LANGUAGES = languages
        tokenizer.TO_LANGUAGE_CODE = language_aliases
        whisper.tokenizer = tokenizer
        whisper.log_mel_spectrogram = log_mel_spectrogram
        sys.modules["whisper"] = whisper
        sys.modules["whisper.tokenizer"] = tokenizer

    if importlib.util.find_spec("wetext") is None:
        if text_frontend:
            raise RuntimeError("text_frontend=1 requires wetext or ttsfrd in the speech image")
        wetext = types.ModuleType("wetext")
        wetext.__spec__ = importlib.machinery.ModuleSpec("wetext", loader=None)
        processing = types.ModuleType("wetext.processing")
        processing.__spec__ = importlib.machinery.ModuleSpec("wetext.processing", loader=None)

        class Normalizer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def normalize(self, text: str, *args: Any, **kwargs: Any) -> str:
                return text

        class TextNormalizer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def normalize(self, text: str, *args: Any, **kwargs: Any) -> list[str]:
                return [text]

        wetext.Normalizer = Normalizer
        processing.TextNormalizer = TextNormalizer
        wetext.processing = processing
        sys.modules["wetext"] = wetext
        sys.modules["wetext.processing"] = processing

    try:
        dataset_package = importlib.import_module("cosyvoice.dataset")
    except Exception:
        dataset_package = None
    install_cosyvoice_dataset_processor_stub(dataset_package, importlib, types)
    install_matcha_inference_adapters(importlib, types)


def install_cosyvoice_dataset_processor_stub(dataset_package: Any, importlib: Any, types: Any) -> None:
    processor = types.ModuleType("cosyvoice.dataset.processor")
    processor.__spec__ = importlib.machinery.ModuleSpec("cosyvoice.dataset.processor", loader=None)

    def unavailable_processor(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("CosyVoice dataset processor is not available in the inference image")

    for name in (
        "parquet_opener",
        "filter",
        "resample",
        "feat_extractor",
        "compute_fbank",
        "parse_embedding",
        "tokenize",
        "shuffle",
        "sort",
        "batch",
        "padding",
    ):
        setattr(processor, name, unavailable_processor)
    sys.modules["cosyvoice.dataset.processor"] = processor
    if dataset_package is not None:
        setattr(dataset_package, "processor", processor)


def install_matcha_inference_adapters(importlib: Any, types: Any) -> None:
    def rank_zero_only(fn: Any = None, *args: Any, **kwargs: Any) -> Any:
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    lightning = types.ModuleType("lightning")
    lightning.__spec__ = importlib.machinery.ModuleSpec("lightning", loader=None)

    class Callback:
        pass

    class LightningDataModule:
        pass

    class LightningModule:
        pass

    class Trainer:
        pass

    lightning.Callback = Callback
    lightning.LightningDataModule = LightningDataModule
    lightning.LightningModule = LightningModule
    lightning.Trainer = Trainer
    sys.modules.setdefault("lightning", lightning)

    pytorch = types.ModuleType("lightning.pytorch")
    pytorch.__spec__ = importlib.machinery.ModuleSpec("lightning.pytorch", loader=None)
    loggers = types.ModuleType("lightning.pytorch.loggers")
    loggers.__spec__ = importlib.machinery.ModuleSpec("lightning.pytorch.loggers", loader=None)
    utilities = types.ModuleType("lightning.pytorch.utilities")
    utilities.__spec__ = importlib.machinery.ModuleSpec("lightning.pytorch.utilities", loader=None)

    class Logger:
        pass

    def grad_norm(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {}

    loggers.Logger = Logger
    utilities.rank_zero_only = rank_zero_only
    utilities.grad_norm = grad_norm
    pytorch.loggers = loggers
    pytorch.utilities = utilities
    lightning.pytorch = pytorch
    sys.modules.setdefault("lightning.pytorch", pytorch)
    sys.modules.setdefault("lightning.pytorch.loggers", loggers)
    sys.modules.setdefault("lightning.pytorch.utilities", utilities)

    matcha_utils = types.ModuleType("matcha.utils")
    matcha_utils.__spec__ = importlib.machinery.ModuleSpec("matcha.utils", loader=None)
    pylogger = types.ModuleType("matcha.utils.pylogger")
    pylogger.__spec__ = importlib.machinery.ModuleSpec("matcha.utils.pylogger", loader=None)
    model = types.ModuleType("matcha.utils.model")
    model.__spec__ = importlib.machinery.ModuleSpec("matcha.utils.model", loader=None)
    audio = types.ModuleType("matcha.utils.audio")
    audio.__spec__ = importlib.machinery.ModuleSpec("matcha.utils.audio", loader=None)

    def get_pylogger(name: str = __name__) -> Any:
        import logging

        return logging.getLogger(name)

    def sequence_mask(length: Any, max_length: int | None = None) -> Any:
        import torch

        if max_length is None:
            max_length = int(length.max().item())
        x = torch.arange(max_length, device=length.device, dtype=length.dtype)
        return x.unsqueeze(0) < length.unsqueeze(1)

    def fix_len_compatibility(length: int, num_downsamplings_in_unet: int = 2) -> int:
        factor = 2**num_downsamplings_in_unet
        while length % factor:
            length += 1
        return length

    def normalize(data: Any, mu: Any, std: Any) -> Any:
        return (data - mu) / std

    def denormalize(data: Any, mu: Any, std: Any) -> Any:
        return data * std + mu

    mel_basis: dict[str, Any] = {}
    hann_window: dict[str, Any] = {}

    def mel_spectrogram(
        y: Any,
        n_fft: int,
        num_mels: int,
        sampling_rate: int,
        hop_size: int,
        win_size: int,
        fmin: int,
        fmax: int | None,
        center: bool = False,
    ) -> Any:
        import torch
        from librosa.filters import mel as librosa_mel

        key = f"{fmax}_{y.device}"
        if key not in mel_basis:
            mel = librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
            hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
        padded = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        ).squeeze(1)
        spec = torch.view_as_real(
            torch.stft(
                padded,
                n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=hann_window[str(y.device)],
                center=center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(mel_basis[key], spec)
        return torch.log(torch.clamp(spec, min=1e-5))

    pylogger.get_pylogger = get_pylogger
    model.sequence_mask = sequence_mask
    model.fix_len_compatibility = fix_len_compatibility
    model.normalize = normalize
    model.denormalize = denormalize
    audio.mel_spectrogram = mel_spectrogram
    matcha_utils.pylogger = pylogger
    matcha_utils.model = model
    matcha_utils.audio = audio
    matcha_utils.get_pylogger = get_pylogger
    try:
        matcha_package = importlib.import_module("matcha")
        setattr(matcha_package, "utils", matcha_utils)
    except Exception:
        pass
    sys.modules["matcha.utils"] = matcha_utils
    sys.modules["matcha.utils.pylogger"] = pylogger
    sys.modules["matcha.utils.model"] = model
    sys.modules["matcha.utils.audio"] = audio


