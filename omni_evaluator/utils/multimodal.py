# OmniEvaluator
# Copyright (c) 2026-present NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import audioread
import av
import base64
import copy
import cv2
import gzip
import io
import ipaddress
import librosa
import logging
import math
import numpy as np
import os
import PIL
from PIL import Image
import pydub
import requests
import socket
import soundfile
import tempfile
import time
import torch
import torchaudio
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Sized, Type
from urllib.parse import urlparse
import wave

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests (connect, read) in seconds
_DEFAULT_REQUEST_TIMEOUT = (5, 60)


_ALLOWED_HOSTS_CACHE: Tuple[Optional[str], frozenset, tuple] = (None, frozenset(), tuple())


def _parse_allowed_hosts(raw: Optional[str]) -> Tuple[frozenset, tuple]:
    # Parse comma-separated ALLOWED_HOSTS into (hostname set, IP network tuple).
    # Entries parseable as IP/CIDR go to networks; the rest are treated as hostnames.
    if not raw:
        return frozenset(), tuple()
    hosts, networks = set(), list()
    for entry in raw.split(","):
        entry = entry.strip().lower()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            hosts.add(entry)
    return frozenset(hosts), tuple(networks)


def _get_allowed_hosts() -> Tuple[frozenset, tuple]:
    # Read ALLOWED_HOSTS env on every call, but cache the parse result keyed on raw string.
    global _ALLOWED_HOSTS_CACHE
    raw = os.environ.get("ALLOWED_HOSTS", "")
    if _ALLOWED_HOSTS_CACHE[0] != raw:
        _ALLOWED_HOSTS_CACHE = (raw, *_parse_allowed_hosts(raw))
    return _ALLOWED_HOSTS_CACHE[1], _ALLOWED_HOSTS_CACHE[2]


def _validate_url_safe(url: str) -> None:
    """Validate that a URL does not point to a private/internal network address (SSRF protection).

    Hosts/CIDRs listed in the ``ALLOWED_HOSTS`` env var (comma-separated) bypass the check.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"No hostname in URL: {url}")
    allowed_hosts, allowed_networks = _get_allowed_hosts()
    if hostname.lower() in allowed_hosts:
        return
    try:
        resolved_ips = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")
    for family, _, _, _, sockaddr in resolved_ips:
        ip = ipaddress.ip_address(sockaddr[0])
        if any(ip in net for net in allowed_networks):
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(
                f"URL resolves to a private/internal address ({ip}), "
                f"which is blocked for security: {url}. "
                f"If this host is trusted, add it to the ALLOWED_HOSTS env var or "
                f"--allowed-hosts CLI arg."
            )


_REQUEST_RETRYABLE_EXCS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.SSLError,
)


def safe_request_get(
    url: str,
    timeout=_DEFAULT_REQUEST_TIMEOUT,
    max_retry: int = 3,
    **kwargs,
) -> requests.Response:
    """Wrapper around requests.get() with SSRF protection, mandatory timeout,
    and bounded retry on transient network errors.

    Retries on ``ChunkedEncodingError`` / ``ConnectionError`` / ``Timeout`` /
    ``SSLError`` with exponential backoff (1s, 2s, 4s, capped at 10s).
    Non-transient failures (HTTP 4xx/5xx via ``raise_for_status``) propagate
    immediately. SSRF validation runs once before the retry loop.
    """
    _validate_url_safe(url)
    _last_ex = None
    for _attempt in range(1, max_retry + 1):
        try:
            response = requests.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except _REQUEST_RETRYABLE_EXCS as ex:
            _last_ex = ex
            if _attempt >= max_retry:
                break
            _backoff = min(2 ** (_attempt - 1), 10)
            logger.warning(
                'safe_request_get(%s) attempt %d/%d failed (%s); retrying in %ds',
                url[:120], _attempt, max_retry, type(ex).__name__, _backoff,
            )
            time.sleep(_backoff)
    # All retries exhausted — re-raise the last transient error.
    raise _last_ex

AudioDecoder = Type
try:
    from datasets.features._torchcodec import AudioDecoder
except Exception as ex:
    logger.warning('Cannot import `datasets.features._torchcodec.AudioDecoder`')
    logger.warning('It is recommended to run after update of dependency `datasets>=4.0.0`')

from omni_evaluator.utils.string import is_url, decode_base64_string


def to_pil_image(
    image: Union[str, bytes, np.ndarray, PIL.Image.Image]
) -> Image.Image:
    if isinstance(image, PIL.Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, str):
        if os.path.exists(image): # image_path
            return Image.open(image)
        elif is_url(image):
            response = safe_request_get(image)
            return Image.open(io.BytesIO(response.content))
        else:
            if ";base64," in image: # remove pattern such as 'data:image/jpeg;base64,'
                image = image.split(";base64,", 1)[1]
            image = decode_base64_string(image)
            # image = base64.b64decode(image)
            image = Image.open(io.BytesIO(image))

    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    return image

def to_image_bytes(
    image: Union[PIL.Image.Image, str, bytes],
    extension: Optional[str] = None,
    encode_base64: bool = True,
) -> Union[bytes, str]:
    detected_extension = extension

    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
        if detected_extension is None:
            detected_extension = image.format or "JPEG"
    elif isinstance(image, str):
        if os.path.exists(image):
            image = Image.open(image)
            if detected_extension is None:
                detected_extension = image.format or "JPEG"
        elif is_url(image):
            response = safe_request_get(image)
            image = Image.open(io.BytesIO(response.content))
            if detected_extension is None:
                detected_extension = image.format or "JPEG"
        else:
            if ";base64," in image: # remove pattern such as 'data:image/jpeg;base64,'
                image = image.split(";base64,", 1)[1]
            image = Image.open(io.BytesIO(decode_base64_string(image)))
            if detected_extension is None:
                detected_extension = image.format or "JPEG"

    if detected_extension is None:
        detected_extension = getattr(image, "format", None) or "JPEG"

    # JPEG does not support alpha channel; only convert to RGB for JPEG output
    if detected_extension.upper() == "JPEG" and image.mode != "RGB":
        image = image.convert("RGB")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format=detected_extension) # save image to bytes
    image_bytes = image_bytes.getvalue()

    if encode_base64:
        image_bytes = base64.standard_b64encode(image_bytes) # encode base64
        image_bytes = image_bytes.decode("utf-8")
    return image_bytes

def resize_image(
    image: PIL.Image.Image, 
    max_height: int, 
    max_width: int,
) -> Image.Image:
    width, height = image.size
    if min(width, height) < 50:
        scale = 50 / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)))
    if (
        width < max_width
        and height < max_height
    ): # do not resize
        return image

    width_scale = max_width / width
    height_scale = max_height / height
    if width_scale < height_scale:
        new_width = int(width * width_scale)
        new_height = int(height * width_scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        new_width = int(width * height_scale)
        new_height = int(height * height_scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def detect_audio_format(audio_bytes: bytes) -> Optional[str]:
    header = audio_bytes[:12]

    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    elif header[:4] == b"fLaC":
        return "flac"
    elif header[:4] == b"OggS":
        return "ogg"
    elif header[:3] == b"ID3" or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    elif header[:4] == b"FORM" and header[8:12] == b"AIFF":
        return "aiff"
    elif header[4:8] == b"ftyp":
        return "m4a"
    elif header[:5] == b"#!AMR":
        return "amr"
    else:
        return None


_PIL_FORMAT_TO_MIME: dict = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}

_AUDIO_FORMAT_TO_MIME: dict = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    "aiff": "audio/aiff",
    "amr": "audio/amr",
}

_VIDEO_FORMAT_TO_MIME: dict = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "m4v": "video/x-m4v",
}


def detect_image_format(image_bytes: bytes) -> Optional[str]:
    """Detect image format from bytes content using PIL.
    Returns PIL format name (e.g. 'JPEG', 'PNG', 'WEBP'), or None if undetectable."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.format
    except Exception:
        return None


def detect_video_format(video_bytes: bytes) -> Optional[str]:
    """Detect video format from magic bytes.
    Returns lowercase format string (e.g. 'mp4', 'webm'), or None if undetectable."""
    if len(video_bytes) < 12:
        return None
    header = video_bytes[:12]
    if header[4:8] == b"ftyp":
        subtype = video_bytes[8:12]
        if subtype in (b"qt  ", b"MOVI"):
            return "mov"
        return "mp4"
    elif header[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    elif header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "avi"
    return None


def image_mime_type(image_bytes: bytes, fallback: str = "image/jpeg") -> str:
    """Get MIME type string for image bytes."""
    fmt = detect_image_format(image_bytes)
    if fmt:
        return _PIL_FORMAT_TO_MIME.get(fmt.upper(), fallback)
    return fallback


def audio_mime_type(audio_bytes: bytes, fallback: str = "audio/wav") -> str:
    """Get MIME type string for audio bytes."""
    fmt = detect_audio_format(audio_bytes)
    if fmt:
        return _AUDIO_FORMAT_TO_MIME.get(fmt.lower(), fallback)
    return fallback


def video_mime_type(video_bytes: bytes, fallback: str = "video/mp4") -> str:
    """Get MIME type string for video bytes."""
    fmt = detect_video_format(video_bytes)
    if fmt:
        return _VIDEO_FORMAT_TO_MIME.get(fmt.lower(), fallback)
    return fallback


def media_mime_type(media_bytes: bytes, fallback: str = "application/octet-stream") -> str:
    """Detect MIME type from media bytes (video, audio, or image), using content not filename."""
    fmt = detect_video_format(media_bytes)
    if fmt:
        return _VIDEO_FORMAT_TO_MIME.get(fmt.lower(), fallback)
    fmt = detect_audio_format(media_bytes)
    if fmt:
        return _AUDIO_FORMAT_TO_MIME.get(fmt.lower(), fallback)
    fmt = detect_image_format(media_bytes)
    if fmt:
        return _PIL_FORMAT_TO_MIME.get(fmt.upper(), fallback)
    return fallback

def to_nparray_audio(
    audio: Union[str, bytes, np.ndarray],
    sampling_rate: Optional[int] = None,
    mono: Optional[bool] = True,
) -> Tuple[np.ndarray, Optional[int], Optional[str]]:
    audio_format = None
    if isinstance(audio, np.ndarray):
        pass
    elif isinstance(audio, (bytes, io.BytesIO)):
        if isinstance(audio, io.BytesIO):
            audio = audio.getvalue()
        audio_format = detect_audio_format(audio_bytes=audio)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=f'.{audio_format}', delete=True) as fp:
            fp.write(audio)
            fp.flush()
            audio, sampling_rate = librosa.load(
                fp.name,
                # audioread.ffdec.FFmpegAudioFile(fp.name),
                sr=sampling_rate,
                mono=mono,
            )
    elif isinstance(audio, str):
        if os.path.exists(audio):
            audio, sampling_rate = librosa.load(
                audio,
                # audioread.ffdec.FFmpegAudioFile(fp.name),
                sr=sampling_rate,
                mono=mono,
            )
        elif is_url(audio):
            response = safe_request_get(audio)
            audio = response.content # bytes
            if isinstance(audio, str):
                if ";base64," in audio: # remove pattern such as 'data:audio/wav;base64,'
                    audio = audio.split(";base64,", 1)[1]
                audio = decode_base64_string(audio) # bytes
            return to_nparray_audio(
                audio=audio,
                sampling_rate=sampling_rate,
                mono=mono,
            )
        else: # base64 encoded string
            if ";base64," in audio: # remove pattern such as 'data:audio/wav;base64,'
                audio = audio.split(";base64,", 1)[1]
            audio = decode_base64_string(audio) # bytes
            return to_nparray_audio(
                audio=audio,
                sampling_rate=sampling_rate,
                mono=mono,
            )
    elif isinstance(audio, AudioDecoder):
        audio = audio.get_all_samples()
        sampling_rate = audio.sample_rate
        audio = audio.data[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
    else:
        raise TypeError(f'Invalid type of audio: {type(audio)}')

    return audio, sampling_rate, audio_format

def to_audio_bytes(
    audio: Union[str, bytes, np.ndarray],
    extension: str = "WAV",
    sampling_rate: int = 16000,
    encode_base64: bool = False,
) -> Union[bytes, str]:
    audio_bytes = None
    if isinstance(audio, np.ndarray):
        with io.BytesIO() as buffer:
            soundfile.write(buffer, audio, sampling_rate, format=extension)
            audio_bytes = buffer.getvalue()
    elif isinstance(audio, bytes):
        audio_bytes = audio
    elif isinstance(audio, str):
        if os.path.exists(audio):
            with open(audio, "rb") as fp:
                audio_bytes = fp.read()
        elif is_url(audio):
            response = safe_request_get(audio)
            audio_bytes = response.content
        else: # base64 encoded string
            if ";base64," in audio: # remove pattern such as 'data:audio/wav;base64,'
                audio = audio.split(";base64,", 1)[1]
            audio_bytes = decode_base64_string(audio)
    elif isinstance(audio, AudioDecoder):
        if isinstance(getattr(audio, "_hf_encoded", None), dict):
            audio_bytes = audio._hf_encoded.get("bytes", None)
        if not isinstance(audio_bytes, bytes):
            audio = audio.get_all_samples()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
                torchaudio.save(
                    fp.name, 
                    audio.data, 
                    sample_rate=audio.sample_rate or sampling_rate, 
                    format="wav",
                ) 
                fp.seek(0)
                audio_bytes = fp.read()
    elif isinstance(audio, dict):
        # Standard keys of the HF datasets.Audio feature:
        #   - Audio(decode=False) yields {"bytes": b"...", "path": "..."}
        #   - Audio(decode=True)  yields {"array": np.ndarray, "sampling_rate": int, "path": "..."}
        # OpenAI / Anthropic style keys: {"value" | "audio" | "input_audio" | "data": ...}
        # priority: bytes (most raw) > array (decoded) > path > openai-style.
        # only array uses an explicit `is not None` check to avoid ndarray bool ambiguity.
        if audio.get("bytes"):
            audio_bytes = to_audio_bytes(audio=audio["bytes"], extension=extension)
        elif audio.get("array") is not None:
            _sr = audio.get("sampling_rate", sampling_rate)
            audio_bytes = to_audio_bytes(audio=audio["array"], extension=extension, sampling_rate=_sr)
        elif audio.get("path"):
            audio_bytes = to_audio_bytes(audio=audio["path"], extension=extension)
        elif audio.get("value", None):
            audio_bytes = to_audio_bytes(audio=audio["value"])
        elif audio.get("audio", None):
            audio_bytes = to_audio_bytes(audio=audio["audio"])
        elif audio.get("input_audio", None):
            audio_bytes = to_audio_bytes(audio=audio["input_audio"])
        elif audio.get("data", None):
            audio_bytes = to_audio_bytes(audio=audio["data"])
    
    audio_format = detect_audio_format(audio_bytes=audio_bytes)
    if (
        not audio_format
        or audio_format.lower() != extension.lower()
    ):
        audio_pydub = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes))
        _buffer = io.BytesIO()
        audio_pydub.export(_buffer, format="wav")
        audio_bytes = _buffer.getvalue()        
    
    if encode_base64:
        audio_bytes = base64.b64encode(audio_bytes)
        audio_bytes = audio_bytes.decode("utf-8")
    return audio_bytes

def to_audio_wav(
    audio: Union[str, bytes, np.ndarray],
    path: str,
    sampling_rate: Optional[int] = 16_000,
    channels: Optional[int] = 1,
) -> str:
    audio_bytes = to_audio_bytes(audio=audio) # audio_bytes
    if (
        audio_bytes[:4] == b"RIFF" 
        and audio_bytes[8:12] == b"WAVE"
    ): # wav bytes
        with open(path, "wb") as fp:
            fp.write(audio_bytes)
    else: # pcm bytes
        with wave.open(path, "wb") as fp:
            fp.setnchannels(channels)
            fp.setsampwidth(2) # int16 = 2bytes
            fp.setframerate(sampling_rate)
            fp.writeframes(audio_bytes)
    return path

def to_video_bytes(
    video: Union[str, bytes],
    encode_base64: bool = False,
    timeout: Optional[int] = 600,
) -> Union[bytes, str]:
    video_bytes = None
    if isinstance(video, bytes):
        video_bytes = video
    elif isinstance(video, (io.BytesIO, bytearray)):
        video_bytes = bytes(video) if isinstance(video, bytearray) else video.getvalue()
    elif isinstance(video, str):
        if os.path.exists(video):
            with open(video, "rb") as fp:
                video_bytes = fp.read()
        elif is_url(video):
            response = safe_request_get(video, timeout=timeout)
            video_bytes = response.content
        else:  # base64 encoded string
            if ";base64," in video: # remove pattern such as 'data:video/mp4;base64,'
                video = video.split(";base64,", 1)[1]
            video_bytes = decode_base64_string(video)
    else:
        raise TypeError(f'Invalid type of video: {type(video)}')

    if encode_base64:
        video_bytes = base64.b64encode(video_bytes)
        video_bytes = video_bytes.decode("utf-8")
    return video_bytes


def to_nparray_video(
    video: Union[str, bytes, np.ndarray],
    max_frames: Optional[int] = 128,
    frame_format: Optional[str] = "rgb24",
    sampling_rate: Optional[int] = 16_000,
    mono: Optional[bool] = True,
    audio_format: Optional[str] = "s16",
    default_time_base: Optional[int] = 1_000_000, # microseconds
    timeout: Optional[int] = 600,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
    container = None
    if isinstance(video, np.ndarray):
        return video, None, None
    
    container = _to_av_container(
        video=video,
        max_frames=max_frames,
        timeout=timeout,
    )
    frames = _extract_frames_from_container(
        container=container,
        max_frames=max_frames,
        frame_format=frame_format,
        default_time_base=default_time_base,
    )
    
    container = _to_av_container(
        video=video,
        max_frames=max_frames,
        timeout=timeout,
    )
    wav, sampling_rate = _extract_wav_from_container(
        container=container,
        sampling_rate=sampling_rate,
        mono=mono,
        audio_format=audio_format,
    )
    return frames, wav, sampling_rate

def _to_av_container(
    video: Union[str, bytes, np.ndarray],
    max_frames: Optional[int] = 128,
    timeout: Optional[int] = 600,
):
    container = None
    if isinstance(video, (bytes, bytearray, io.BytesIO)):
        if isinstance(video, io.BytesIO):
            video = video.getvalue()
        container = av.open(io.BytesIO(video))
    elif hasattr(video, "read"):
        container = av.open(video)
    elif isinstance(video, str):
        if os.path.exists(video): # video_path
            container = av.open(video)
        elif is_url(video): # video_url
            _response = safe_request_get(video, timeout=timeout)
            video = _response.content
            if isinstance(video, str):
                if video.startswith("data:video/"): # remove pattern such as 'data:video/jpeg;base64,'
                    video = video.split(",")[1]
            return _to_av_container(
                video=video,
                max_frames=max_frames,
                timeout=timeout,
            )
        else: # base64 encoded string
            video = decode_base64_string(video) # bytes
            return _to_av_container(
                video=video,
                max_frames=max_frames,
                timeout=timeout,
            )
    else:
        raise TypeError(f'Invalid type of video: {type(video)}')

    return container

def _extract_frames_from_container(
    container: av.container.input.InputContainer,
    max_frames: Optional[int] = 128,
    frame_format: Optional[str] = "rgb24",
    default_time_base: Optional[int] = 1_000_000, # microseconds
):
    # extract video frames
    frames = list()
    for _stream in tqdm(
        container.streams.video, 
        initial=0, 
        total=len(container.streams.video),
        desc=f'Extract frames from video',
    ): 
        _duration_seconds = _stream.duration * _stream.time_base
        _duration_seconds = float(_duration_seconds)
        
        _time_steps = np.linspace(0.0, max(_duration_seconds - 1e-3, 0.0), num=int(_duration_seconds))
        # uniform sampling if max_frames given
        if max_frames: 
            _time_steps = np.linspace(0.0, max(_duration_seconds - 1e-3, 0.0), num=max_frames)

        for _time_step in tqdm(
            _time_steps,
            initial=0,
            total=len(_time_steps),
            desc=f'Extracting frames from stream',
        ):
            # seek in microseconds
            # backward=True indicates to move right before keyframe 
            # (in order to decode the most closest frame after move)
            try:
                container.seek(
                    int(_time_step * default_time_base),
                    any_frame=False, 
                    backward=True,
                )
            except Exception:
                # continue if seek failed
                continue

            # decode only first frame after seek
            for _frame in container.decode(_stream):
                if (
                    _frame.time is not None 
                    and _frame.time + 1e-6 < _time_step
                ): # if frame.time is too older than _time_step, skip to next frame
                    continue

                _frame = _frame.to_ndarray(format=frame_format)
                frames.append(_frame)
                got = True
                break
        
        break # usually use only first stream

    # (T, H, W, 3)
    frames = np.stack(frames, axis=0)
    return frames

def _extract_wav_from_container(
    container: av.container.input.InputContainer,
    sampling_rate: Optional[int] = 16_000,
    mono: Optional[bool] = True,
    audio_format: Optional[str] = "s16",
):
    # extract audio wav
    chunks = list()
    for _stream in tqdm(
        container.streams.audio, 
        initial=0, 
        total=len(container.streams.audio),
        desc=f'Extract streams from audio',
    ):
        _layout = "mono"
        if not mono:
            if _stream.layout:
                _layout = _stream.layout.name
            else:
                _layout = "stereo"

        _resampler = av.audio.resampler.AudioResampler(
            format=audio_format,
            layout=_layout,
            rate=sampling_rate,
        )

        for _frame in tqdm(
            container.decode(_stream),
            desc=f'Extracting audio wav from stream',
        ):
            _frames = _resampler.resample(_frame)
            for _frame_ in _frames:
                if _frame_ is None:
                    continue
                _chunk = _frame_.to_ndarray() # (C, T)
                _chunk = _chunk[0] if _chunk.ndim == 2 else _chunk
                chunks.append(_chunk.astype(np.int16, copy=False))

    wav = None
    if not sampling_rate:
        sampling_rate = _resampler.rate
    if chunks:
        pcm16 = np.concatenate(chunks)
        wav = (pcm16.astype(np.float32) / 32768.0).astype(np.float32)

    return wav, sampling_rate

def count_video_frames(
    path: str,
) -> int:
    container = av.open(path)
    stream = container.streams.video[0]
    num_frames = 0
    for _ in container.decode(stream):
        num_frames += 1
    container.close()
    return num_frames