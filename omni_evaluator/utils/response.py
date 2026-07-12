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

"""HTTP response diagnostics — surface payload shape on 4xx without leaking values.

When an inference / chat-completion endpoint returns a 4xx, the upstream error
message is usually too generic to tell *which* content field has an
unexpected shape (a list/dict in a string slot, a missing key, a malformed
``image_url`` block, ...). Reproducing the call is expensive (large media in
the messages), so the diagnostic must run at the moment of failure.

This module logs the *type and keys only* — never the values themselves —
so base64 images, audio bytes, and other large or sensitive media never
leak to stdout. The 4xx branch in each engine's chat-completion wrapper
calls :func:`summarize_payload_shape` right before raising.

Envelopes vary across engines:
- OpenAI-compatible (vllm, sglang chat, openai, anthropic):
  ``messages_key="messages"``, ``content_key="content"`` (defaults).
- Gemini / google: ``messages_key="contents"``, ``content_key="parts"``.
- Plain ``completions`` endpoint (no message list): pass ``messages_key=None``;
  only the top-level field shapes and any media URL prefixes are reported.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Mapping, Optional


_DEFAULT_MEDIA_URL_TYPES: tuple = ("video_url", "image_url", "audio_url")


def _shape(value: Any) -> str:
    """One-line type/shape description that never dumps the value itself."""
    if isinstance(value, dict):
        return "dict(" + ", ".join(
            f"{_k}={type(_v).__name__}" for _k, _v in value.items()
        ) + ")"
    if isinstance(value, list):
        if not value:
            return "list(empty)"
        return f"list[{type(value[0]).__name__}]({len(value)})"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    return type(value).__name__


def _content_block_shape(block: Any) -> Any:
    """Single content block — type / keys / value_types (no values)."""
    if not isinstance(block, Mapping):
        return type(block).__name__
    return {
        "type": block.get("type"),
        "keys": sorted(block.keys()),
        "value_types": {
            _k: _shape(_v) for _k, _v in block.items() if _k != "type"
        },
    }


def _message_shape(msg: Mapping, content_key: str) -> Dict[str, Any]:
    content = msg.get(content_key)
    if isinstance(content, list):
        content_repr: Any = [_content_block_shape(_b) for _b in content]
    else:
        content_repr = type(content).__name__
    return {"role": msg.get("role"), content_key: content_repr}


def summarize_payload_shape(
    payload: Mapping,
    *,
    status: int,
    logger: logging.Logger,
    messages_key: Optional[str] = "messages",
    content_key: str = "content",
    media_url_types: Iterable[str] = _DEFAULT_MEDIA_URL_TYPES,
    url_prefix_chars: int = 60,
    max_chars: int = 3000,
) -> None:
    """Log payload type/keys (no values) for HTTP 4xx-style diagnosis.

    Three sections are emitted, each as a single ``logger.error`` line so the
    output stays grep-friendly. If reflection fails partway, the function logs
    ``payload preview dump failed`` (with traceback) and returns — the caller
    can still raise its own error.

    Args:
        payload:           request body about to be sent (or as serialized
                           by the engine before the SDK call).
        status:            HTTP status code, included in the log prefix.
        logger:            caller's logger.
        messages_key:      envelope key holding the message list. ``None``
                           skips the per-message section (e.g. ``completions``
                           endpoints that take ``prompt`` instead).
        content_key:       per-message content key. Default ``"content"``
                           (OpenAI-compatible); pass ``"parts"`` for Gemini.
        media_url_types:   content-block ``type`` values whose ``.url`` prefix
                           is informative (http / file / s3-presigned / data:).
        url_prefix_chars:  truncate the logged media URL to this many chars so
                           a base64 ``data:`` URI doesn't flood the log.
        max_chars:         ``json.dumps`` cap per section, to keep one log line
                           readable.
    """
    try:
        if messages_key is not None:
            messages = payload.get(messages_key, []) or []
            preview = [
                _message_shape(_m, content_key)
                if isinstance(_m, Mapping) else type(_m).__name__
                for _m in messages
            ]
            logger.error(
                'payload preview (HTTP %d diag) %s: %s',
                status, messages_key,
                json.dumps(preview, default=str)[:max_chars],
            )

        # Top-level fields (model / tools / chat_template_kwargs /
        # mm_processor_kwargs / generation kwargs ...) — any list/dict here is a
        # likely culprit when the messages list itself looks well-formed.
        top: Dict[str, str] = {}
        for _k, _v in payload.items():
            if _k == messages_key:
                continue
            if isinstance(_v, str):
                top[_k] = f"str[:40]={_v[:40]!r}"
            else:
                top[_k] = _shape(_v)
        logger.error(
            'payload preview (HTTP %d diag) top-level: %s',
            status, json.dumps(top, default=str)[:max_chars],
        )

        if messages_key is not None:
            media_url_types = tuple(media_url_types)
            for _m in payload.get(messages_key, []) or []:
                if not isinstance(_m, Mapping):
                    continue
                for _block in (_m.get(content_key) or []):
                    if not isinstance(_block, Mapping):
                        continue
                    _btype = _block.get("type")
                    if _btype in media_url_types:
                        _url = (_block.get(_btype) or {}).get("url")
                        logger.error(
                            'payload preview (HTTP %d diag) %s.url[:%d]: %r',
                            status, _btype, url_prefix_chars,
                            str(_url)[:url_prefix_chars],
                        )
    except Exception:
        logger.error('payload preview dump failed', exc_info=True)


def is_valid_response(output: Mapping[str, Any]) -> bool:
    """True when a normalized ``parse_response`` output carries usable content.

    A 200-OK response that produced no text and no tool/function call is a
    *content-contract* failure (safety block / refusal, or a reasoning model that
    spent the whole output budget on thinking). Re-sending the identical request
    cannot change the outcome, so provider engines fail-fast on this instead of
    burning ``max_retry`` attempts. (Transient failures raise exceptions and are
    handled by the retry/backoff branch instead.)

    This operates on the *provider-agnostic* fields produced by every engine's
    ``parse_response`` (``prediction`` / ``generated_text`` / ``tool_calls`` /
    ``function_call`` / ``error_message``), so all providers share one rule rather
    than each re-inspecting its own raw response shape.
    """
    if not isinstance(output, Mapping):
        return False
    if output.get("error_message"):
        return False
    if output.get("tool_calls") or output.get("function_call"):
        return True
    # Usable when any text field carries non-empty text or a non-empty structured
    # value. ``generated_text`` is a list of candidate strings; ``prediction`` may
    # be str / dict / number — walk both, descending into nested lists.
    _stack = [output.get("prediction"), output.get("generated_text")]
    while _stack:
        _value = _stack.pop()
        if isinstance(_value, str):
            if _value.strip():
                return True
        elif isinstance(_value, (list, tuple)):
            _stack.extend(_value)
        elif _value:  # dict / number — a non-empty structured answer counts as usable
            return True
    return False
