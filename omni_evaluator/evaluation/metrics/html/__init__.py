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

"""HTML tree-edit-distance utility (zss + bs4 + nltk).

Public utilities consumed by ``TextEvaluator.compute_tree_edit`` /
``compute_tree_edit_score``:
  - ``parse_html`` / ``parse_html_table`` / ``parse_html_math``: pre-parsers
  - ``remove_html_tag``: drop inline-style tags
  - ``create_node`` / ``create_html_tree``: zss tree construction
  - ``_get_insert_cost_html`` / ``_get_remove_cost_html`` / ``_get_update_cost_html``: edit costs
  - ``generate_html``: optional stage-1 LLM conversion (pred → HTML/MathML)

Distance + score computation (legacy ``_get_dist`` + ``get_score``) lives in
``TextEvaluator.compute_html_tree_distance`` + ``compute_tree_edit`` — same
sub-module split as ``mmmu_accuracy`` / ``wtq`` / ``pier`` / ``repetition``.

Stage-1 prompt strings live in ``prompts.py`` as module constants; yaml
selects them by ``source_format`` so callers don't paste long multi-line strings.
"""
import asyncio
import re
from typing import Dict, List, Optional, Union

import zss
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from nltk import edit_distance
from zss import Node

from omni_evaluator.api.chat_completions import (
    batch_chat_completion_async,
    batch_chat_completion_sync,
)
from omni_evaluator.evaluation.metrics.html.prompts import PROMPTS, PromptSet
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
)


MAX_DEPTH = 100

# Same list as legacy: inline-style tags that don't carry structural meaning.
REMOVE_TAGS = [
    "br", "a", "b", "strong", "i", "em", "u", "ins", "s", "del",
    "sub", "sup", "q", "blockquote", "cite", "mark", "code",
    "samp", "kbd", "var", "pre",
]


# ─── pre-parsers ───────────────────────────────────────────────────────────
def parse_html(text: str) -> str:
    """Strip surrounding ```html ... ``` fence if present (legacy)."""
    html = text
    idx = html.find("```html")
    html = html[idx + len("```html"):] if idx >= 0 else ""
    idx = html.find("```")
    html = html[:idx] if idx >= 0 else html
    html = html.strip()
    html = html if len(html) > 0 else text
    return html


def parse_html_table(html: str) -> str:
    soup = BeautifulSoup(html, "html5lib")
    tables = soup.find_all("table")
    tables = [t for t in tables if t.find_parent("table") is None]
    return "\n".join(str(t).strip() for t in tables)


def parse_html_math(html: str) -> str:
    soup = BeautifulSoup(html, "html5lib")
    maths = soup.find_all("math")
    maths = [m for m in maths if m.find_parent("math") is None]
    return "\n".join(str(m).strip() for m in maths)


def remove_html_tag(html: str) -> str:
    for tag in REMOVE_TAGS:
        html = re.sub(rf"<{tag}(\s+[^\s]*?)*>", "", html)
        html = re.sub(rf"<{tag}(\s+[^\s]*?)*/>", "", html)
        html = re.sub(rf"</{tag}>", "", html)
    return html


# ─── tree construction ────────────────────────────────────────────────────
def create_node(label: str, **attrs: Dict) -> Node:
    node = Node(label)
    node.attrs = attrs
    return node


def create_html_tree(
    elem: Union[str, BeautifulSoup, Tag, NavigableString],
    max_depth: int = MAX_DEPTH,
) -> Optional[Node]:
    node = None
    if max_depth <= 0:
        return node
    if isinstance(elem, BeautifulSoup):
        node = create_html_tree(elem.body, max_depth)
    elif isinstance(elem, Tag):
        children = list(elem.children)
        children = [create_html_tree(c, max_depth - 1) for c in children]
        children = [c for c in children if c is not None]
        rowspan = elem.get("rowspan", "1")
        colspan = elem.get("colspan", "1")
        node = create_node("", rowspan=rowspan, colspan=colspan)
        for c in children:
            node.addkid(c)
        if len(node.children) == 1:
            if node.label == node.children[0].label and node.attrs == node.children[0].attrs:
                node = node.children[0]
    if isinstance(elem, Comment):
        pass
    elif isinstance(elem, NavigableString):
        text = str(elem)
        text = re.sub(r"\s+", "", text)
        node = create_node(text) if len(text) > 0 else None
    elif isinstance(elem, str):
        soup = BeautifulSoup(elem, "html5lib")
        node = create_html_tree(soup, max_depth)
    return node


# ─── zss cost functions ───────────────────────────────────────────────────
def _get_insert_cost_html(node: Node) -> float:
    return 1.0


def _get_remove_cost_html(node: Node) -> float:
    return 1.0


def _get_update_cost_html(node1: Node, node2: Node) -> float:
    if node1.attrs == node2.attrs:
        return edit_distance(node1.label, node2.label) / max(len(node1.label), len(node2.label), 1)
    return 1.0


# ─── stage-1 LLM conversion (optional) ────────────────────────────────────
def generate_html(
    predictions: List[str],
    *,
    source_format: str,
    api_name: str,
    groups: Optional[List[Optional[str]]] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    do_async: bool = False,
    semaphore_size: int = 4,
) -> List[str]:
    """Batch convert ``predictions`` (LaTeX/Markdown) → HTML (or MathML).

    Empty predictions pass through unchanged (skipped from the LLM call).
    Per-sample ``group`` is consulted only when the selected ``source_format``
    has ``equation_*`` prompts and ``group == "equation"`` (legacy latex
    equation-vs-text branching).

    Args:
        predictions: per-sample raw prediction string.
        source_format: key into ``PROMPTS`` (e.g. ``"latex"``, ``"markdown"``).
        api_name: chat completion endpoint (passed to
            ``batch_chat_completion_sync``/``_async``).
        groups: per-sample group label (None if absent).
        max_tokens / temperature: LLM generation options.
        do_async: use the async client + ``semaphore_size`` if True.

    Returns:
        List of converted strings, aligned 1:1 to ``predictions``.
    """
    if source_format not in PROMPTS:
        raise KeyError(
            f"Unknown source_format={source_format!r}; available: {list(PROMPTS)}"
        )
    _prompts: PromptSet = PROMPTS[source_format]
    if groups is None:
        groups = [None] * len(predictions)

    _messages_list: List[List[ChatMessage]] = []
    _generation_options_list: List[Dict] = []
    _call_indices: List[int] = []
    for _idx, (_prediction, _group) in enumerate(zip(predictions, groups)):
        if not _prediction:
            continue
        if (
            _prompts.equation_user_prompt_template is not None
            and _group == "equation"
        ):
            _system_prompt = _prompts.equation_system_prompt or _prompts.system_prompt
            _user_prompt = _prompts.equation_user_prompt_template.format(pred=_prediction)
        else:
            _system_prompt = _prompts.system_prompt
            _user_prompt = _prompts.user_prompt_template.format(pred=_prediction)
        _messages_list.append([
            ChatMessage(role="system", content=[
                ChatTextContent(type="text", value=_system_prompt),
            ]),
            ChatMessage(role="user", content=[
                ChatTextContent(type="text", value=_user_prompt),
            ]),
        ])
        _generation_options_list.append({"max_tokens": max_tokens, "temperature": temperature})
        _call_indices.append(_idx)

    if not _messages_list:
        return list(predictions)

    if do_async:
        _responses = asyncio.run(batch_chat_completion_async(
            api_name=api_name,
            messages_list=_messages_list,
            generation_options_list=_generation_options_list,
            semaphore_size=semaphore_size,
        ))
    else:
        _responses = batch_chat_completion_sync(
            api_name=api_name,
            messages_list=_messages_list,
            generation_options_list=_generation_options_list,
        )

    _converted = list(predictions)
    for _call_idx, _record_idx in enumerate(_call_indices):
        _response = _responses[_call_idx]
        _converted_html = (
            _response["prediction"] if isinstance(_response, dict)
            else getattr(_response, "prediction", None)
        )
        _converted[_record_idx] = _converted_html or ""
    return _converted


