from __future__ import annotations

import os
import re
import struct
import threading
import time
from typing import Any

from app.common import env_bool, env_float, env_value, log

try:
    import usb.core
    import usb.util
except ImportError:  # pragma: no cover - handled at runtime in slim environments
    usb = None  # type: ignore[assignment]


PARAMETERS = {
    "AGCONOFF": (19, 0, "int", "rw"),
    "ECHOONOFF": (19, 14, "int", "rw"),
    "NLATTENONOFF": (19, 18, "int", "rw"),
    "VOICEACTIVITY": (19, 32, "int", "ro"),
    "STATNOISEONOFF_SR": (19, 33, "int", "rw"),
    "NONSTATNOISEONOFF_SR": (19, 34, "int", "rw"),
    "GAMMA_NS_SR": (19, 35, "float", "rw"),
    "GAMMA_NN_SR": (19, 36, "float", "rw"),
    "GAMMAVAD_SR": (19, 39, "float", "rw"),
    "DOAANGLE": (21, 0, "int", "ro"),
}

DIRECTION_SECTORS = (
    ("front", "正前方"),
    ("front_right", "右前方"),
    ("right", "右侧"),
    ("back_right", "右后方"),
    ("back", "正后方"),
    ("back_left", "左后方"),
    ("left", "左侧"),
    ("front_left", "左前方"),
)

DIRECTION_QUERY_RE = re.compile(
    r"(我在你(的)?(哪边|哪一边|哪儿|哪里|什么方向|哪个方向|左边|右边|前面|后面)|"
    r"你(能|可以)?(听出|判断|知道)?我在(哪边|哪一边|哪儿|哪里|什么方向|哪个方向))"
)


def env_int_base(key: str) -> int:
    try:
        return int(env_value(key), 0)
    except ValueError:
        raise RuntimeError(f"{key} must be an integer in runtime.env") from None


def env_float_default(key: str, default: str) -> float:
    raw_value = os.getenv(key, default)
    value = raw_value.strip() or default
    try:
        return float(value)
    except ValueError:
        raise RuntimeError(f"{key} must be a number in runtime.env") from None


def direction_sector(angle: int) -> dict[str, str]:
    code, label = DIRECTION_SECTORS[int(((angle + 22.5) % 360) // 45)]
    return {"code": code, "label": label}


def direction_label(angle: int | float) -> str:
    return direction_sector(int(round(angle)) % 360)["label"]


def is_direction_query(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？、,.!?]", "", text).replace("您", "你")
    normalized = re.sub(r"(哪|那){2,}", r"\1", normalized)
    return bool(DIRECTION_QUERY_RE.search(normalized))


class ReSpeakerAudioSource:
    TIMEOUT_MS = 2000

    def __init__(self, dev: Any) -> None:
        self.dev = dev
        self.front_offset_degrees = env_float("RESPEAKER_DOA_FRONT_OFFSET_DEGREES")
        self.clockwise = env_bool("RESPEAKER_DOA_CLOCKWISE")
        self._lock = threading.Lock()

    def close(self) -> None:
        if usb is not None:
            with self._lock:
                usb.util.dispose_resources(self.dev)

    def read(self, name: str) -> int | float:
        parameter = PARAMETERS[name]
        parameter_id, offset, value_type, _ = parameter
        command = 0x80 | int(offset)
        if value_type == "int":
            command |= 0x40
        with self._lock:
            response = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                command,
                int(parameter_id),
                8,
                self.TIMEOUT_MS,
            )
        raw = bytes(response)
        integer, exponent = struct.unpack("ii", raw)
        if value_type == "int":
            return integer
        return integer * (2.0**exponent)

    def write(self, name: str, value: int | float) -> None:
        parameter = PARAMETERS[name]
        parameter_id, offset, value_type, mode = parameter
        if mode != "rw":
            raise ValueError(f"{name} is read-only")
        if value_type == "int":
            payload = struct.pack("iii", int(offset), int(value), 1)
        else:
            payload = struct.pack("ifi", int(offset), float(value), 0)
        with self._lock:
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                0,
                int(parameter_id),
                payload,
                self.TIMEOUT_MS,
            )

    def apply_tuning(self) -> None:
        if not env_bool("RESPEAKER_TUNING_ENABLED"):
            return

        writes: tuple[tuple[str, int | float], ...] = (
            ("AGCONOFF", int(env_bool("RESPEAKER_AGC_ENABLED"))),
            ("STATNOISEONOFF_SR", int(env_bool("RESPEAKER_ASR_STATIONARY_NOISE_SUPPRESSION"))),
            ("NONSTATNOISEONOFF_SR", int(env_bool("RESPEAKER_ASR_NONSTATIONARY_NOISE_SUPPRESSION"))),
            ("GAMMA_NS_SR", env_float("RESPEAKER_ASR_STATIONARY_NOISE_SUPPRESSION_LEVEL")),
            ("GAMMA_NN_SR", env_float("RESPEAKER_ASR_NONSTATIONARY_NOISE_SUPPRESSION_LEVEL")),
            ("GAMMAVAD_SR", env_float("RESPEAKER_VAD_THRESHOLD_DB")),
            ("ECHOONOFF", int(env_bool("RESPEAKER_ECHO_SUPPRESSION_ENABLED"))),
            ("NLATTENONOFF", int(env_bool("RESPEAKER_NONLINEAR_AEC_ENABLED"))),
        )
        for name, value in writes:
            try:
                self.write(name, value)
            except Exception as exc:
                log(f"respeaker tuning write failed: {name}={value}: {exc}", level="warning")
        log("respeaker tuning applied")

    def normalize_angle(self, raw_angle: int) -> int:
        angle = (raw_angle - self.front_offset_degrees) % 360
        if not self.clockwise:
            angle = (-angle) % 360
        return int(round(angle)) % 360

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        try:
            raw_angle = int(self.read("DOAANGLE"))
            angle = self.normalize_angle(raw_angle)
        except Exception as exc:
            return {
                "ok": False,
                "source": "respeaker",
                "error": str(exc),
                "updated_at": now,
            }

        try:
            voice_activity: bool | None = bool(self.read("VOICEACTIVITY"))
        except Exception as exc:
            log(f"respeaker voice activity read failed: {exc}", level="debug")
            voice_activity = None

        sector = direction_sector(angle)
        return {
            "ok": True,
            "source": "respeaker",
            "raw_angle_degrees": raw_angle,
            "angle_degrees": angle,
            "sector": sector["code"],
            "label": sector["label"],
            "voice_activity": voice_activity,
            "coordinate": {
                "zero": "front",
                "positive": "clockwise",
                "unit": "degrees",
                "front_offset_degrees": self.front_offset_degrees,
                "device_clockwise": self.clockwise,
            },
            "updated_at": now,
        }

    def answer_direction(self) -> str:
        snapshot = self.snapshot()
        if not snapshot.get("ok"):
            return "我现在读不到麦克风方向信息。"
        return f"您在我的{snapshot['label']}。"


def _open_respeaker_once(*, log_missing: bool = True) -> ReSpeakerAudioSource | None:
    vendor_id = env_int_base("RESPEAKER_USB_VENDOR_ID")
    product_id = env_int_base("RESPEAKER_USB_PRODUCT_ID")
    try:
        dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
    except Exception as exc:
        if log_missing:
            log(f"respeaker control unavailable: {exc}", level="warning")
        return None
    if dev is None:
        if log_missing:
            log(f"respeaker control device not found: vid=0x{vendor_id:04x} pid=0x{product_id:04x}")
        return None

    source = ReSpeakerAudioSource(dev)
    source.apply_tuning()
    snapshot = source.snapshot()
    log(f"respeaker control ready: {snapshot}")
    return source


class ReSpeakerAudioSourceManager:
    def __init__(self, retry_seconds: float = 5.0) -> None:
        self.retry_seconds = max(1.0, retry_seconds)
        self._source: ReSpeakerAudioSource | None = None
        self._lock = threading.Lock()
        self._next_open_at = 0.0
        self._last_missing_log_at = -30.0

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    def _open_locked(self) -> ReSpeakerAudioSource | None:
        if self._source is not None:
            return self._source

        now = time.monotonic()
        if now < self._next_open_at:
            return None

        log_missing = now - self._last_missing_log_at >= 30.0
        source = _open_respeaker_once(log_missing=log_missing)
        if source is None:
            if log_missing:
                self._last_missing_log_at = now
            self._next_open_at = now + self.retry_seconds
            return None

        self._source = source
        self._next_open_at = 0.0
        self._last_missing_log_at = 0.0
        return source

    def _drop_source(self, source: ReSpeakerAudioSource, reason: str) -> None:
        with self._lock:
            if self._source is source:
                log(f"respeaker control lost; will retry: {reason}", level="warning")
                self._close_locked()
                self._next_open_at = time.monotonic() + self.retry_seconds

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            source = self._open_locked()
        if source is None:
            return {
                "ok": False,
                "source": "respeaker",
                "error": "unavailable",
                "updated_at": time.time(),
            }

        snapshot = source.snapshot()
        if not snapshot.get("ok"):
            self._drop_source(source, str(snapshot.get("error") or "read failed"))
        return snapshot

    def answer_direction(self) -> str:
        snapshot = self.snapshot()
        if not snapshot.get("ok"):
            return "我现在读不到麦克风方向信息。"
        return f"您在我的{snapshot['label']}。"


def open_respeaker() -> ReSpeakerAudioSourceManager | None:
    if not env_bool("RESPEAKER_ENABLED"):
        log("respeaker control disabled")
        return None
    if usb is None:
        log("respeaker control unavailable: pyusb is not installed", level="warning")
        return None
    return ReSpeakerAudioSourceManager(env_float_default("RESPEAKER_RECONNECT_SECONDS", "5"))


def direction_answer(text: str, source: ReSpeakerAudioSource | ReSpeakerAudioSourceManager | None) -> str | None:
    if not is_direction_query(text):
        return None
    if source is None:
        return "我现在读不到麦克风方向信息。"
    return source.answer_direction()


def direction_answer_from_snapshot(text: str, snapshot: dict[str, Any]) -> str | None:
    if not is_direction_query(text):
        return None
    if not snapshot.get("ok"):
        return "我现在读不到麦克风方向信息。"
    angle = snapshot.get("angle_degrees")
    if angle is None:
        return "我现在读不到麦克风方向信息。"
    try:
        label = direction_label(float(angle))
    except (TypeError, ValueError):
        return "我现在读不到麦克风方向信息。"
    return f"您在我的{label}。"
