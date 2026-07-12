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

"""Build inference_output-style data from evaluation outputs."""


# Multimodal message parts carry their payload (often a base64 blob) in
# "value"; only text parts should contribute to the displayed question.
_MEDIA_PART_TYPES = {"image", "image_url", "video", "video_url", "audio", "audio_url"}


def message_text_from_messages(messages: list) -> str:
    if not isinstance(messages, list):
        return ""

    def _part_text(part) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            # Skip image/audio/video parts so their base64 payload never leaks
            # into the question text (consistent with _extract_media_from_messages).
            if str(part.get("type") or "").lower() in _MEDIA_PART_TYPES:
                return ""
            for key in ("value", "text", "content"):
                v = part.get(key)
                if isinstance(v, str):
                    return v
        return ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = [_part_text(p) for p in content]
            joined = "\n".join([c for c in chunks if c])
            if joined:
                return joined
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = [_part_text(p) for p in content]
            joined = "\n".join([c for c in chunks if c])
            if joined:
                return joined
    return ""


def build_inference_output_from_eval(data: dict) -> dict | None:
    """Build inference_output-style JSON from evaluation output when only inference list exists."""
    inf_list = data.get("inference")
    if not isinstance(inf_list, list) or not inf_list:
        return None
    if len(inf_list) == 1 and isinstance(inf_list[0], list):
        inf_list = inf_list[0]
    elif inf_list and all(isinstance(x, list) for x in inf_list):
        merged: list = []
        for chunk in inf_list:
            if isinstance(chunk, list):
                merged.extend(chunk)
        inf_list = merged
    out_list: list[dict] = []
    for item in inf_list:
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            prompt = ""
        messages = item.get("messages") if isinstance(item.get("messages"), list) else []
        question = item.get("question")
        if not isinstance(question, str):
            question = ""
        if not question:
            alt = item.get("input") or item.get("query")
            if isinstance(alt, str):
                question = alt
        if not question and prompt:
            question = prompt
        if not question:
            question = message_text_from_messages(messages)
        pred = item.get("prediction_postprocessed")
        if pred is None:
            pred = item.get("prediction")
        if pred is None:
            pred = item.get("answer")
        if pred is None:
            pred = ""
        label = item.get("label")
        gt = item.get("ground_truth")
        if gt is None:
            gt = label
        if gt is None:
            gt = ""
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        rec = {
            "question": question,
            "prompt": prompt or question,
            "ground_truth": gt,
            "answer": gt,
            "prediction": pred,
            "metrics": metrics,
            "options": item.get("options") or item.get("option_contents") or [],
            "messages": messages,
            "meta": item.get("meta") if isinstance(item.get("meta"), dict) else {},
            "index": item.get("index"),
        }
        out_list.append(rec)
    if not out_list:
        return None
    return {"output": out_list}
