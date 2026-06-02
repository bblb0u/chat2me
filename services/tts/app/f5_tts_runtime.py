from __future__ import annotations

import functools
import importlib.machinery
import importlib.resources
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.runtime import log


def install_f5_runtime_shims() -> None:
    if not hasattr(functools, "cache"):
        functools.cache = functools.lru_cache(maxsize=None)  # type: ignore[attr-defined]

    if not hasattr(importlib.resources, "files"):
        from importlib_resources import files

        importlib.resources.files = files  # type: ignore[attr-defined]

    install_encodec_stub()
    install_torchaudio_shim()
    install_f5_trainer_stub()


def install_encodec_stub() -> None:
    if "encodec" in sys.modules:
        return

    encodec = types.ModuleType("encodec")
    encodec.__spec__ = importlib.machinery.ModuleSpec("encodec", loader=None)

    class EncodecModel:
        @staticmethod
        def encodec_model_24khz(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("encodec is not available in the F5-TTS inference runtime")

        @staticmethod
        def encodec_model_48khz(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("encodec is not available in the F5-TTS inference runtime")

    encodec.EncodecModel = EncodecModel
    sys.modules["encodec"] = encodec


def install_f5_trainer_stub() -> None:
    module_name = "f5_tts.model.trainer"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)

    class Trainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("F5-TTS training is not available in the speech inference image")

    module.Trainer = Trainer
    sys.modules[module_name] = module


def install_torchaudio_shim() -> None:
    if "torchaudio" in sys.modules:
        return

    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    torchaudio = types.ModuleType("torchaudio")
    transforms = types.ModuleType("torchaudio.transforms")
    compliance = types.ModuleType("torchaudio.compliance")
    kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    functional_pkg = types.ModuleType("torchaudio.functional")
    functional = types.ModuleType("torchaudio.functional.functional")

    torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchaudio.transforms", loader=None)
    compliance.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance", loader=None)
    kaldi.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance.kaldi", loader=None)
    functional_pkg.__spec__ = importlib.machinery.ModuleSpec("torchaudio.functional", loader=None)
    functional.__spec__ = importlib.machinery.ModuleSpec("torchaudio.functional.functional", loader=None)

    class MelSpectrogram(torch.nn.Module):
        def __init__(
            self,
            sample_rate: int = 16000,
            n_fft: int = 400,
            win_length: int | None = None,
            hop_length: int | None = None,
            n_mels: int = 128,
            power: float = 2.0,
            center: bool = True,
            normalized: bool = False,
            norm: str | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.win_length = win_length or n_fft
            self.hop_length = hop_length or self.win_length // 2
            self.n_mels = n_mels
            self.power = power
            self.center = center
            self.normalized = normalized
            self.norm = norm
            self.register_buffer("_empty", torch.tensor(0.0), persistent=False)

        def forward(self, waveform: Any) -> Any:
            from librosa.filters import mel as librosa_mel

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            waveform = waveform.to(dtype=torch.float32)
            window = torch.hann_window(self.win_length, device=waveform.device, dtype=waveform.dtype)
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                pad_mode="reflect",
                normalized=self.normalized,
                onesided=True,
                return_complex=True,
            ).abs()
            if self.power != 1:
                spec = spec.pow(self.power)
            basis = librosa_mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                norm=self.norm,
            )
            mel_basis = torch.from_numpy(basis).to(device=waveform.device, dtype=waveform.dtype)
            return torch.matmul(mel_basis, spec)

    class Resample(torch.nn.Module):
        def __init__(self, orig_freq: int, new_freq: int, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.orig_freq = int(orig_freq)
            self.new_freq = int(new_freq)

        def forward(self, waveform: Any) -> Any:
            if self.orig_freq == self.new_freq:
                return waveform
            device = waveform.device
            dtype = waveform.dtype
            array = waveform.detach().cpu().numpy()
            gcd = math.gcd(self.orig_freq, self.new_freq)
            resampled = resample_poly(array, self.new_freq // gcd, self.orig_freq // gcd, axis=-1)
            return torch.from_numpy(np.asarray(resampled, dtype=np.float32)).to(device=device, dtype=dtype)

    def load(path: str | Path, *args: Any, **kwargs: Any) -> tuple[Any, int]:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = torch.from_numpy(data.T.copy())
        return audio, int(sample_rate)

    def save(path: str | Path, tensor: Any, sample_rate: int, *args: Any, **kwargs: Any) -> None:
        array = tensor.detach().cpu().numpy()
        if array.ndim == 2:
            array = array.T
        sf.write(str(path), array, sample_rate)

    def hz_to_mel(freq: Any, mel_scale: str = "htk") -> Any:
        value = torch.as_tensor(freq, dtype=torch.float32)
        return 2595.0 * torch.log10(1.0 + value / 700.0)

    def mel_to_hz(mels: Any, mel_scale: str = "htk") -> Any:
        value = torch.as_tensor(mels, dtype=torch.float32)
        return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

    transforms.MelSpectrogram = MelSpectrogram
    transforms.Resample = Resample
    functional._hz_to_mel = hz_to_mel
    functional._mel_to_hz = mel_to_hz
    kaldi.fbank = _missing_torchaudio_compliance
    compliance.kaldi = kaldi
    functional_pkg.functional = functional
    torchaudio.transforms = transforms
    torchaudio.compliance = compliance
    torchaudio.functional = functional_pkg
    torchaudio.load = load
    torchaudio.save = save

    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.transforms"] = transforms
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.compliance.kaldi"] = kaldi
    sys.modules["torchaudio.functional"] = functional_pkg
    sys.modules["torchaudio.functional.functional"] = functional


def _missing_torchaudio_compliance(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("torchaudio.compliance is not available in this inference image")


def package_file(package: str, relative_path: str) -> Path:
    from importlib_resources import files

    return Path(str(files(package).joinpath(relative_path)))


def load_audio_mono(path: Path, sample_rate: int) -> Any:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    data, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    samples = np.mean(data, axis=1).astype(np.float32, copy=False)
    if int(source_rate) != int(sample_rate):
        gcd = math.gcd(int(source_rate), int(sample_rate))
        samples = resample_poly(samples, int(sample_rate) // gcd, int(source_rate) // gcd).astype(np.float32)
    return torch.from_numpy(samples.copy()).unsqueeze(0)


def load_f5_checkpoint(model: Any, ckpt_path: Path, device: str, fp16: bool, use_ema: bool) -> Any:
    import torch

    dtype = torch.float16 if fp16 and device.startswith("cuda") else torch.float32
    if fp16 and not device.startswith("cuda"):
        log("F5-TTS fp16 requested but CUDA is unavailable for this runtime; using fp32")
    model = model.to(dtype=dtype)

    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(ckpt_path), device="cpu")
        if use_ema:
            state_dict = {
                key.replace("ema_model.", ""): value
                for key, value in checkpoint.items()
                if key not in {"initted", "step"}
            }
        else:
            state_dict = checkpoint
    else:
        try:
            checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        if use_ema:
            checkpoint = checkpoint["ema_model_state_dict"]
            state_dict = {
                key.replace("ema_model.", ""): value
                for key, value in checkpoint.items()
                if key not in {"initted", "step"}
            }
        else:
            state_dict = checkpoint.get("model_state_dict", checkpoint)

    for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        state_dict.pop(key, None)
    model.load_state_dict(state_dict)
    del checkpoint, state_dict
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return model.to(device).eval()


def load_vocos(vocoder_dir: Path, device: str) -> Any:
    import torch
    from vocos import Vocos

    config_path = vocoder_dir / "config.yaml"
    model_path = vocoder_dir / "pytorch_model.bin"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing required file: {config_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"missing required file: {model_path}")

    vocoder = Vocos.from_hparams(str(config_path))
    try:
        state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(str(model_path), map_location="cpu")
    for key in ("feature_extractor.mel_spec.mel_scale.fb", "feature_extractor.mel_spec.spectrogram.window"):
        state_dict.pop(key, None)
    vocoder.load_state_dict(state_dict)
    return vocoder.eval().to(device)


class F5TTSRuntime:
    def __init__(
        self,
        *,
        model_name: str,
        model_dir: Path,
        ckpt_file: Path,
        vocoder_dir: Path,
        ref_audio: Path | None,
        ref_text: str,
        device: str,
        fp16: bool,
        use_ema: bool,
        ode_method: str,
        nfe_step: int,
        cfg_strength: float,
        sway_sampling_coef: float,
        speed: float,
        target_rms: float,
        seed: int | None,
    ) -> None:
        install_f5_runtime_shims()

        import torch
        from f5_tts.model.backbones.dit import DiT
        from f5_tts.model.cfm import CFM
        from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

        if model_name != "F5TTS_v1_Base":
            raise RuntimeError("F5-TTS engine currently supports F5TTS_v1_Base")
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"missing required file: {ckpt_file}")

        started = time.monotonic()
        config_file = model_dir / "config.yaml"
        if not config_file.is_file():
            config_file = package_file("f5_tts", f"configs/{model_name}.yaml")
        with config_file.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        model_config = config["model"]
        mel_config = model_config["mel_spec"]
        model_arc = dict(model_config["arch"])
        self.sample_rate = int(mel_config["target_sample_rate"])
        self.n_mel_channels = int(mel_config["n_mel_channels"])
        self.hop_length = int(mel_config["hop_length"])
        self.mel_spec_type = str(mel_config["mel_spec_type"])
        if self.mel_spec_type != "vocos":
            raise RuntimeError("F5-TTS engine currently supports the vocos mel backend")

        self.device = resolve_torch_device(device)
        self.fp16 = fp16 and self.device.startswith("cuda")
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.speed = speed
        self.target_rms = target_rms
        self.seed = seed
        self.convert_char_to_pinyin = convert_char_to_pinyin

        vocab_file = model_dir / "vocab.txt"
        if not vocab_file.is_file():
            vocab_file = package_file("f5_tts", "infer/examples/vocab.txt")
        vocab_char_map, vocab_size = get_tokenizer(str(vocab_file), "custom")

        self.vocoder = load_vocos(vocoder_dir, self.device)
        self.model = CFM(
            transformer=DiT(**model_arc, text_num_embeds=vocab_size, mel_dim=self.n_mel_channels),
            mel_spec_kwargs={
                "n_fft": int(mel_config["n_fft"]),
                "hop_length": self.hop_length,
                "win_length": int(mel_config["win_length"]),
                "n_mel_channels": self.n_mel_channels,
                "target_sample_rate": self.sample_rate,
                "mel_spec_type": self.mel_spec_type,
            },
            odeint_kwargs={"method": ode_method},
            vocab_char_map=vocab_char_map,
        )
        self.model = load_f5_checkpoint(self.model, ckpt_file, self.device, self.fp16, use_ema)

        if ref_audio is None:
            ref_audio = package_file("f5_tts", "infer/examples/basic/basic_ref_zh.wav")
        if not ref_audio.is_file():
            raise FileNotFoundError(f"missing required F5-TTS reference audio: {ref_audio}")
        if not ref_text.strip():
            raise RuntimeError("F5_TTS_REF_TEXT must be set; empty ref text would trigger an extra ASR model")
        self.ref_text = normalize_ref_text(ref_text)
        self.ref_audio = load_audio_mono(ref_audio, self.sample_rate)
        self.ref_rms = torch.sqrt(torch.mean(torch.square(self.ref_audio))).item()
        if self.ref_rms < self.target_rms:
            self.ref_audio = self.ref_audio * self.target_rms / max(self.ref_rms, 1e-6)
        self.ref_audio = self.ref_audio.to(self.device)
        self.ref_audio_frames = int(self.ref_audio.shape[-1] // self.hop_length)
        log(
            "F5-TTS loaded: "
            f"model={model_name} device={self.device} fp16={self.fp16} "
            f"sample_rate={self.sample_rate} nfe={self.nfe_step} "
            f"elapsed={time.monotonic() - started:.2f}s"
        )

    def generate(self, text: str) -> np.ndarray:
        import torch

        gen_text = text.strip()
        if not gen_text:
            return np.zeros(0, dtype=np.float32)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(self.seed)

        ref_bytes = max(1, len(self.ref_text.encode("utf-8")))
        gen_bytes = max(1, len(gen_text.encode("utf-8")))
        duration = self.ref_audio_frames + int(self.ref_audio_frames / ref_bytes * gen_bytes / max(self.speed, 0.05))
        duration = max(self.ref_audio_frames + 1, duration)
        text_tokens = self.convert_char_to_pinyin([self.ref_text + gen_text])
        started = time.monotonic()
        with torch.inference_mode():
            generated, _ = self.model.sample(
                cond=self.ref_audio,
                text=text_tokens,
                duration=duration,
                steps=self.nfe_step,
                cfg_strength=self.cfg_strength,
                sway_sampling_coef=self.sway_sampling_coef,
            )
            generated = generated.to(torch.float32)[:, self.ref_audio_frames :, :].permute(0, 2, 1)
            waveform = self.vocoder.decode(generated).squeeze().detach().cpu().numpy().astype(np.float32)
        if self.ref_rms < self.target_rms:
            waveform = waveform * self.ref_rms / max(self.target_rms, 1e-6)
        log(
            "F5-TTS synthesized: "
            f"text_chars={len(gen_text)} duration={len(waveform) / max(1, self.sample_rate):.2f}s "
            f"elapsed={time.monotonic() - started:.2f}s"
        )
        return waveform


def normalize_ref_text(text: str) -> str:
    normalized = text.strip()
    if not normalized.endswith((". ", "。")):
        normalized = normalized + (" " if normalized.endswith(".") else "。")
    return normalized


def resolve_torch_device(requested: str) -> str:
    import torch

    value = requested.strip().lower() or "auto"
    if value in {"auto", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("F5-TTS requires CUDA for VOICE_TTS_DEVICE=cuda, but torch CUDA is unavailable")
        return "cuda"
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"F5-TTS requires CUDA for VOICE_TTS_DEVICE={requested}, but torch CUDA is unavailable")
        device_index = value.split(":", 1)[1]
        if not device_index.isdigit():
            raise RuntimeError("VOICE_TTS_DEVICE must be auto, cpu, cuda, gpu, or cuda:<index> for F5-TTS")
        if int(device_index) >= torch.cuda.device_count():
            raise RuntimeError(
                f"VOICE_TTS_DEVICE={requested} is not available; torch sees {torch.cuda.device_count()} CUDA device(s)"
            )
        return value
    if value == "cpu":
        return "cpu"
    raise RuntimeError("VOICE_TTS_DEVICE must be auto, cpu, cuda, gpu, or cuda:<index> for F5-TTS")
