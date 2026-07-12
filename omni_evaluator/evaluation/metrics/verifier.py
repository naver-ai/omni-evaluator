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

"""Reference-based verifier judge.

``Verifier`` inherits :class:`JudgeEvaluator` (reusing its stateless parse/collect
helpers) and composes a :class:`HuggingfaceInferencer` for the local backend, freed in
``__del__``. The prompt is always ``prompts/verifier.py`` (CoT variant iff ``reasoning``);
the score is parsed from the last line-anchored ``Rating: 0|1``; ``engine`` selects the local
HuggingFace route, a llama.cpp GGUF route, or an ``api/*`` chat-completion route.
"""
import asyncio
from collections import defaultdict
import logging
import multiprocessing
import queue
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from omni_evaluator.api import get_api_group
from omni_evaluator.api.chat_completions import chat_completion_sync, chat_completion_async
from omni_evaluator.evaluation.metrics.judge_evaluator import JudgeEvaluator
from omni_evaluator.evaluation.metrics.prompts.verifier import (
    VERIFIER_PROMPT, VERIFIER_COT_PROMPT,
)
from omni_evaluator.enums.engine import InferenceEngine
from omni_evaluator.inference import TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
)
from omni_evaluator.schemas.generation_options import ApiGenerationOptions, HuggingfaceGenerationOptions
from omni_evaluator.schemas.inference import Record
from omni_evaluator.utils.cpu import get_available_cpu_count, get_numa_node_cpus
from omni_evaluator.utils.io import resolve_gguf_path
from omni_evaluator.utils.resource import split_resources
from omni_evaluator.utils.string import sanitize_name
from omni_evaluator.utils.torch import is_cpu, torch_num_threads

logger = logging.getLogger(__name__)

# The rating digit must end its line (``\s*$``, MULTILINE): the prompt's format-instruction line
# literally contains "'Rating: 0' or 'Rating: 1'", so a model that echoes / repeats the prompt
# (e.g. an under-trained or mis-loaded checkpoint) would otherwise make the last-match parse latch
# onto that trailing "Rating: 1" and score everything 1 even with an empty explanation.
_RATING_RE = re.compile(r"[Rr]ating:\s*([01])\s*$", re.MULTILINE)
_API_ENGINES = (InferenceEngine.api__openai, InferenceEngine.api__anthropic, InferenceEngine.api__google)

# llama.cpp backend: default quant is Q8_0 (overridable via gguf_filename); the glob resolves
# the GGUF inside a local dir / hub repo. The KV context window is tied to max_seq_len (the
# prompt+output budget the fields are truncated to; see _resolve_num_context_tokens) so it is
# exactly large enough and no bigger. _LLAMA_CPP_NUM_CONTEXT_TOKENS is the fallback when max_seq_len
# is unset: 0 -> llama.cpp uses the model's full trained context (Qwen3 = 40960), which never
# overflows but wastes KV.
_GGUF_FILENAME = "*Q8_0.gguf"
_LLAMA_CPP_NUM_CONTEXT_TOKENS = 0

# Budget truncation (mirrors verifier_train/data.py): keep each field's head+tail, drop the
# middle, so the prompt + generated output fit ``max_seq_len`` — matching how the verifier's
# long [Model Answer] was trimmed during training.
_TRUNC_MARGIN = 64          # token headroom left after fitting the fields
_TRUNC_HEAD_RATIO = 0.3     # fraction of a truncated field kept as head (rest kept as tail)


# Sentinel a worker puts on the result queue once it finishes its shard, so the parent can tell
# "shard complete" apart from a per-sample ``(index, text)`` result.
_LLAMA_CPP_WORKER_DONE = "__verifier_llama_cpp_worker_done__"


def _llama_cpp_shard_worker(model_path, num_context_tokens, num_threads, node_cpus, indices, payloads, result_queue):
    """Worker process for the llama.cpp multiprocessing path: pin to a NUMA node, load its own
    Llama, run its record shard, and stream each ``(index, text)`` result on ``result_queue`` as
    it is produced (so the parent's progress bar / verbose logging advance live), then a
    ``_LLAMA_CPP_WORKER_DONE`` sentinel to mark the shard complete. The CPU budget is already
    split across workers, so decode/prefill share the same per-worker thread count."""
    import os
    from llama_cpp import Llama
    try:
        os.sched_setaffinity(0, set(node_cpus))
    except OSError:
        pass
    model = Llama(model_path=model_path, n_ctx=num_context_tokens, n_gpu_layers=0,
                  n_threads=num_threads, n_threads_batch=num_threads, verbose=False)
    for _idx, _payload in zip(indices, payloads):
        _response = model.create_chat_completion(**_payload)
        try:
            _text = _response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            _text = None
        result_queue.put((_idx, _text))
    result_queue.put(_LLAMA_CPP_WORKER_DONE)


class _LlamaTokenizer:
    """Minimal HF-tokenizer-like wrapper over a llama.cpp model (callable -> input_ids, and
    decode), so the shared budget truncation (:meth:`Verifier._truncate_to_budget`) works identically
    for the llama_cpp and huggingface engines."""

    def __init__(self, llama):
        self._llama = llama

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=False)}

    def decode(self, ids, skip_special_tokens=True):
        return self._llama.detokenize(list(ids)).decode("utf-8", errors="ignore")


class Verifier(JudgeEvaluator):
    """A reference-based verifier judge instance (see module docstring)."""

    def __init__(
        self,
        engine: str,
        model_name_or_path: Optional[str] = None,
        api_name: Optional[str] = None,
        reasoning: Union[str, bool, None] = False,
        torch_dtype: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        cache_dir: Optional[str] = None,
        model_group: Optional[str] = "qwen2_omni",
        gguf_filename: Optional[str] = None,
        alias: Optional[str] = None,
        num_concurrency: int = 1,
        num_cpu_threads: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        verbose: bool = False,
        lang: str = "en",
        reason_format: Optional[Union[str, Dict[str, Any]]] = "[REASON]",
        timeout: Optional[Union[int, float]] = None,
        max_retry: Optional[int] = None,
        wait_between_retry: Optional[Union[int, float]] = None,
    ) -> None:
        self.engine = engine
        self.model_name_or_path = model_name_or_path
        self.api_name = api_name
        self.reasoning = reasoning
        self.torch_dtype = torch_dtype
        # device_map is HF-only (None -> HuggingfaceInferencer default); gguf_filename is the
        # llama_cpp quant selector. Keep each field meaningful per engine: llama_cpp runs its own
        # GGUF runtime (device via n_gpu_layers, not device_map) -> force device_map to cpu and keep
        # gguf_filename; every other engine ignores gguf_filename -> drop it. Handled here (not only
        # in VerifierArgs.validate) so the per-task ``verifier:`` override path is covered too.
        if engine == InferenceEngine.llama_cpp.value:
            self.device_map = "cpu"
            self.gguf_filename = gguf_filename
        else:
            self.device_map = device_map
            self.gguf_filename = None
        self.cache_dir = cache_dir
        self.model_group = model_group or "qwen2_omni"
        # label namespacing the metric as verifier_score/{alias}; None -> derived in _verifier_alias.
        self.alias = alias
        # Degree of parallelism, per engine: HF-GPU -> model.generate batch size; api/* ->
        # asyncio.Semaphore in-flight requests; llama_cpp -> worker processes (capped at NUMA
        # node count). HF-CPU ignores it (always sequential bs=1).
        self.num_concurrency = max(1, int(num_concurrency or 1))
        # torch intra-op threads during HF-CPU inference (restored after); None -> leave as-is.
        self.num_cpu_threads = num_cpu_threads
        # token budget for prompt fields (huggingface / llama_cpp); None -> no truncation.
        self.max_seq_len = max_seq_len
        self.verbose = bool(verbose)
        self.lang = lang
        self.reason_format = reason_format or "[REASON]"
        self.timeout = timeout if isinstance(timeout, (int, float)) else TIMEOUT
        self.max_retry = max_retry if isinstance(max_retry, int) else MAX_RETRY
        self.wait_between_retry = (
            wait_between_retry if isinstance(wait_between_retry, (int, float)) else WAIT_BETWEEN_RETRY
        )
        self.huggingface_inferencer = None  # composed HF backend; lazily loaded
        self.llama_cpp_model = None         # llama.cpp GGUF model; lazily loaded
        self.llama_cpp_tokenizer = None     # vocab-only llama.cpp handle for budget truncation

    def _ensure_inferencer(self):
        """Lazily build the composed HuggingfaceInferencer (HF engine only)."""
        if self.huggingface_inferencer is None:
            from omni_evaluator.inference.huggingface.engine import HuggingfaceInferencer
            logger.info(
                "Verifier: loading HF backend %s (group=%s, dtype=%s, device_map=%s, reasoning=%s)",
                self.model_name_or_path, self.model_group, self.torch_dtype,
                self.device_map, self.reasoning,
            )
            self.huggingface_inferencer = HuggingfaceInferencer(
                evaluation_engine="builtin",
                model_name_or_path=self.model_name_or_path,
                model_group=self.model_group,
                reasoning=self.reasoning,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
        return self.huggingface_inferencer

    def _resolve_num_context_tokens(self) -> int:
        """llama.cpp KV context window (its ``n_ctx``). Tie it to max_seq_len (the prompt+output
        budget the fields are truncated to) so the window is exactly large enough; fall back to
        _LLAMA_CPP_NUM_CONTEXT_TOKENS (0 -> the model's full trained context) when truncation is
        disabled."""
        num_context_tokens = _LLAMA_CPP_NUM_CONTEXT_TOKENS
        if self.max_seq_len:
            num_context_tokens = self.max_seq_len
        return num_context_tokens

    def _ensure_llama_cpp_model(self):
        """Lazily load the llama.cpp GGUF model (llama_cpp engine only).

        ``model_name_or_path`` is a local file/dir (see :func:`resolve_gguf_path`) or a HF hub
        repo id (``Llama.from_pretrained`` fetches it). CPU-only (``n_gpu_layers=0``); llama.cpp
        uses its own thread pool (``num_cpu_threads``), so no torch thread scope is needed.
        """
        if self.llama_cpp_model is None:
            from llama_cpp import Llama
            _filename = self.gguf_filename or _GGUF_FILENAME
            _num_context_tokens = self._resolve_num_context_tokens()
            # decode is memory-bandwidth bound (num_cpu_threads, tested ~16); prefill is compute
            # bound, so give it the full CPU budget via n_threads_batch.
            _kwargs = dict(n_ctx=_num_context_tokens, n_gpu_layers=0,
                           n_threads_batch=get_available_cpu_count(), verbose=False)
            if self.num_cpu_threads and self.num_cpu_threads > 0:
                _kwargs["n_threads"] = self.num_cpu_threads
            logger.info(
                "Verifier: loading llama.cpp model %s (filename=%s, n_ctx=%s, n_threads=%s, n_threads_batch=%s)",
                self.model_name_or_path, _filename,
                ("from-model" if _num_context_tokens == 0 else _num_context_tokens),
                _kwargs.get("n_threads"), _kwargs["n_threads_batch"],
            )
            _path = resolve_gguf_path(self.model_name_or_path, _filename)
            if _path is not None:
                self.llama_cpp_model = Llama(model_path=_path, **_kwargs)
            else:
                self.llama_cpp_model = Llama.from_pretrained(
                    repo_id=self.model_name_or_path, filename=_filename, **_kwargs,
                )
        return self.llama_cpp_model

    def _ensure_tokenizer(self):
        """Return an HF-tokenizer-like object for budget truncation (huggingface uses the loaded
        inferencer's tokenizer; llama_cpp uses a cheap vocab-only Llama). None when unavailable
        (api engine, or a hub-only llama_cpp model with no local file)."""
        if self.engine == InferenceEngine.huggingface:
            module = self._ensure_inferencer().module
            return module.tokenizer if module.tokenizer is not None else module.processor
        if self.engine == InferenceEngine.llama_cpp:
            if self.llama_cpp_tokenizer is None:
                _path = resolve_gguf_path(self.model_name_or_path, self.gguf_filename or _GGUF_FILENAME)
                if _path is None:
                    return None
                from llama_cpp import Llama
                self.llama_cpp_tokenizer = _LlamaTokenizer(
                    Llama(model_path=_path, vocab_only=True, verbose=False))
            return self.llama_cpp_tokenizer
        return None

    def __del__(self):
        # Free the GPU on drop (same convention as HuggingfaceInferencer.__del__).
        try:
            if getattr(self, "huggingface_inferencer", None) is not None:
                del self.huggingface_inferencer
                self.huggingface_inferencer = None
            # llama.cpp frees its native context in Llama.__del__; drop the refs.
            self.llama_cpp_model = None
            self.llama_cpp_tokenizer = None
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _build_choices_block(options, option_contents) -> str:
        """Render an ``[Options]`` block for the prompt's ``{choices}`` placeholder
        (``label: content`` per line, bare labels if contents are missing), or ``''``."""
        if not options:
            return ""
        _opts = list(options)
        _lines = ["", "", "[Options]"]              # leading blank line + header
        if option_contents and len(option_contents) == len(_opts):
            for _o, _c in zip(_opts, option_contents):
                _lines.append(f"{_o}: {_c}")
        else:
            _lines.append(", ".join(str(_o) for _o in _opts))
        return "\n".join(_lines)

    @classmethod
    def _truncate_to_budget(cls, tokenizer, reference: str, prediction: str, question: str,
                    budget: int, head_ratio: float = _TRUNC_HEAD_RATIO):
        """Truncate reference / prediction / question (head+tail, middle dropped) so their COMBINED
        tokens fit ``budget`` — bounds the prompt no matter which field is huge. question keeps
        ~20% of the budget; the rest is split between reference and prediction (prediction keeps
        head_ratio/tail so the final answer survives). Mirrors verifier_train/data.py."""
        def _truncate(text, field_budget, ratio, marker):
            # keep head (ratio) + tail, drop the middle
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= field_budget:
                return text
            if field_budget <= 16:                    # essentially no room -> keep a tiny tail
                return tokenizer.decode(ids[-16:], skip_special_tokens=True)
            marker_len = len(tokenizer(marker, add_special_tokens=False)["input_ids"])
            avail = max(2, field_budget - marker_len)
            head = max(1, int(avail * ratio))
            tail = max(1, avail - head)
            return (tokenizer.decode(ids[:head], skip_special_tokens=True) + marker
                    + tokenizer.decode(ids[-tail:], skip_special_tokens=True))

        count = lambda _s: len(tokenizer(_s, add_special_tokens=False)["input_ids"])
        ref_len, pred_len, q_len = count(reference), count(prediction), count(question)
        if budget <= 0 or ref_len + pred_len + q_len <= budget:
            return reference, prediction, question
        q_budget = min(q_len, max(8, int(budget * 0.2)))     # question: small share
        rest = max(2, budget - q_budget)
        ref_budget = min(ref_len, rest // 2)
        pred_budget = min(pred_len, rest - ref_budget)
        ref_budget = min(ref_len, rest - pred_budget)        # give any leftover to reference
        if q_len > q_budget:
            question = _truncate(question, q_budget, 0.5, "\n...[question truncated]...\n")
        if ref_len > ref_budget:
            reference = _truncate(reference, ref_budget, 0.5, "\n...[reference truncated]...\n")
        if pred_len > pred_budget:
            prediction = _truncate(prediction, pred_budget, head_ratio, "\n...[prediction truncated]...\n")
        return reference, prediction, question

    def _format_verifier_prompt(self, record: Union[Dict[str, Any], Record],
                                max_new_tokens: Optional[int] = None) -> str:
        """Fill the verifier prompt (CoT variant iff ``reasoning``) from the record: deduped label
        list, last-user query, and options block. When ``max_seq_len`` + ``max_new_tokens`` are set
        (huggingface / llama_cpp), reference / prediction / question are budget-truncated so the
        prompt + generated output fit ``max_seq_len`` (matching the training-time truncation)."""
        tpl = VERIFIER_COT_PROMPT if self.reasoning else VERIFIER_PROMPT
        user_messages = ChatMessage.get_user_messages(messages=record["messages"])
        query = ChatMessage.get_query(message=user_messages[-1]) if user_messages else ""
        label = record.get("label")
        if isinstance(label, (list, tuple)):
            _seen: set = set()
            label = "\n".join(
                _s for _s in (str(_x).strip() for _x in label if _x is not None)
                if _s and not (_s in _seen or _seen.add(_s))
            )
        prediction = record.get("prediction")
        choices = self._build_choices_block(record.get("options"), record.get("option_contents"))
        if self.max_seq_len and max_new_tokens:
            tokenizer = self._ensure_tokenizer()
            if tokenizer is not None:
                # measure the shell with the real choices block so it is counted in the budget
                # (otherwise a large options list can push the prompt past max_seq_len).
                shell = tpl.format(reference="", prediction="", question="", choices=choices)
                budget = max(0, self.max_seq_len
                             - len(tokenizer(shell, add_special_tokens=False)["input_ids"])
                             - int(max_new_tokens) - _TRUNC_MARGIN)
                label, prediction, query = self._truncate_to_budget(
                    tokenizer, str(label), str(prediction), str(query), budget)
        return tpl.format(
            reference=label,
            prediction=prediction,
            question=query,
            choices=choices,
        )

    def _build_messages(
        self,
        record: Union[Dict[str, Any], Record],
        judge_message: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[ChatMessage]:
        """One user turn holding the verifier prompt (``judge_message`` wins if given;
        ``max_new_tokens`` drives budget truncation of the prompt fields when max_seq_len is set)."""
        text = judge_message or self._format_verifier_prompt(record, max_new_tokens=max_new_tokens)
        return [ChatMessage(role="user", content=[ChatTextContent(type="text", value=text)])]

    def _build_generation_options(self, generation_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize a raw options dict (temperature / max_new_tokens) into the
        engine-specific dict via the shared GenerationOptions schema."""
        _raw = dict(generation_options or {})
        if self.engine == InferenceEngine.huggingface:
            return HuggingfaceGenerationOptions.from_dict(obj=_raw).to_dict()
        if self.engine == InferenceEngine.llama_cpp:
            # create_chat_completion: temperature (0 -> greedy) + max_tokens
            return {"temperature": _raw.get("temperature") or 0.0,
                    "max_tokens": _raw.get("max_tokens", _raw.get("max_new_tokens"))}
        # api/* — provider-normalized (max_new_tokens -> max_tokens)
        _obj = {"temperature": _raw.get("temperature"),
                "max_tokens": _raw.get("max_tokens", _raw.get("max_new_tokens"))}
        return ApiGenerationOptions.from_dict(
            api_name=self.api_name, obj=_obj, api_group=get_api_group(api_name=self.api_name),
        ).to_dict()

    @staticmethod
    def _to_llama_chat(messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Flatten verifier ChatMessages into OpenAI-style string-content chat for llama.cpp."""
        return [
            {"role": _message.role,
             "content": "\n".join(
                 getattr(_content, "value", "") for _content in _message.content
                 if getattr(_content, "type", None) == "text")}
            for _message in messages
        ]

    def _generate(self, messages: List[ChatMessage], generation_options: Dict[str, Any]) -> Optional[str]:
        if self.engine == InferenceEngine.huggingface:
            inferencer = self._ensure_inferencer()
            output = inferencer(
                messages=[_message.to_dict(template="hf") for _message in messages],
                generation_options=generation_options,
                evaluation_method="generation",
                output_modality=["text"],
            )
            if hasattr(output, "prediction"):
                return output.prediction
            if isinstance(output, dict):
                return output.get("prediction")
            return None

        if self.engine == InferenceEngine.llama_cpp:
            model = self._ensure_llama_cpp_model()
            output = model.create_chat_completion(
                messages=self._to_llama_chat(messages), **generation_options)
            try:
                return output["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return None

        # api/* engine
        return chat_completion_sync(
            api_name=self.api_name,
            messages=messages,
            generation_options=generation_options,
            timeout=self.timeout, max_retry=self.max_retry,
            wait_between_retry=self.wait_between_retry,
        )

    def _postprocess(self, response: Optional[str], record=None, idx: Optional[int] = None) -> Dict[str, Any]:
        """Parse one response into a result dict. When verbose (and a ``record`` is given),
        also print the row live via tqdm.write, so per-sample output interleaves with the
        progress bar instead of dumping only after the whole run finishes."""
        rating = None
        if isinstance(response, str):
            _m = _RATING_RE.findall(response)
            if _m:
                rating = int(_m[-1])
        accuracy = float(rating) if rating in (0, 1) else None
        reasons = None
        if self.reason_format and response:
            try:
                reasons = self._parse_response_reason(response=response, rubrics=None)
            except Exception as ex:  # noqa: BLE001
                logger.debug("verifier reason parse failed: %s", ex)
        result = {"accuracy": accuracy, "scores": rating, "reasons": reasons, "response": response}
        if self.verbose and record is not None:
            user_messages = ChatMessage.get_user_messages(messages=record.get("messages") or [])
            query = ChatMessage.get_query(message=user_messages[-1]) if user_messages else ""
            tqdm.write(f'\t[verifier #{idx} | engine={self.engine}]')
            tqdm.write(f'\t- {"Query":<12}: {query}')
            tqdm.write(f'\t- {"Reference":<12}: {record.get("label")}')
            tqdm.write(f'\t- {"Prediction":<12}: {record.get("prediction")}')
            # the assembled prompt actually fed to the model (untruncated) — uncomment to confirm
            # reference / prediction land in the prompt as expected.
            # tqdm.write(f'\t- {"Prompt":<12}: {self._format_verifier_prompt(record)}')
            # full raw generation, before parsing — so one can see what the model actually emitted
            # (e.g. explanation on the next line, or degenerate/echoed output).
            tqdm.write(f'\t- {"Generated":<12}: {response}')
            tqdm.write(f'\t- {"Rating":<12}: {result.get("scores")}')
            tqdm.write(f'\t- {"Explanation":<12}: {result.get("reasons")}')
            tqdm.write(f'\n')
        return result

    def _verifier_alias(self) -> str:
        """Metric-key segment for this verifier (``verifier_score/{alias}``). Explicit alias wins;
        else the api model name (api engines) or the model id (local). For CLI runs VerifierArgs
        pre-derives a compact, sanitized alias; direct construction falls back to the raw model id
        (pass ``alias=`` for a short label). Sanitized so it stays a single safe key segment."""
        alias = self.alias
        if not (isinstance(alias, str) and alias.strip()):
            alias = self.api_name if self.engine in _API_ENGINES else self.model_name_or_path
        return sanitize_name(alias or self.engine)

    def _progress_desc(self, num_workers: int = 1) -> str:
        """Uniform tqdm description across engines: engine name + effective parallelism degree
        (HF-GPU batch size / llama.cpp worker count / api in-flight requests; 1 when sequential)."""
        return f"Evaluating verifier ({self.engine}, x{num_workers})"

    def _judge_huggingface(self, records, prompts, generation_options, show_progress) -> List[Dict[str, Any]]:
        """HF backend. CPU: one record at a time (bs=1; batching hurts on CPU) with intra-op
        threads capped to num_cpu_threads (restored after). GPU: one model.generate per
        num_concurrency chunk via the adapter's batched_generate_text."""
        max_new_tokens = generation_options.get("max_tokens") or generation_options.get("max_new_tokens")
        if is_cpu(self.device_map):
            with torch_num_threads(self.num_cpu_threads):
                judge_results = []
                for _idx, (_record, _prompt) in enumerate(tqdm(list(zip(records, prompts)),
                                             desc=self._progress_desc(1), disable=not show_progress)):
                    judge_results.append(self._postprocess(self._generate(
                        self._build_messages(_record, _prompt, max_new_tokens=max_new_tokens),
                        generation_options), _record, _idx))
                return judge_results
        module = self._ensure_inferencer().module
        judge_results = []
        for _start in tqdm(range(0, len(records), self.num_concurrency),
                           desc=self._progress_desc(self.num_concurrency), disable=not show_progress):
            _messages_list = [
                [_message.to_dict(template="hf")
                 for _message in self._build_messages(_record, _prompt, max_new_tokens=max_new_tokens)]
                for _record, _prompt in zip(records[_start:_start + self.num_concurrency],
                                            prompts[_start:_start + self.num_concurrency])
            ]
            for _j, _output in enumerate(module.batched_generate_text(messages_list=_messages_list,
                                                        generation_options=generation_options)):
                _text = _output.prediction if hasattr(_output, "prediction") else (
                    _output.get("prediction") if isinstance(_output, dict) else _output)
                _idx = _start + _j
                judge_results.append(self._postprocess(_text, records[_idx], _idx))
        return judge_results

    def _judge_llama_cpp(self, records, prompts, generation_options, show_progress) -> List[Dict[str, Any]]:
        """llama.cpp backend. Runs ``num_concurrency`` worker processes, capped at the NUMA node
        count (single-box CPU inference is memory-bandwidth bound); a single worker runs in-process.
        Each worker pins to a NUMA node and loads its own GGUF; records are sharded across them."""
        max_new_tokens = generation_options.get("max_tokens") or generation_options.get("max_new_tokens")

        def _sequential():
            _results = []
            for _idx, (_record, _prompt) in enumerate(tqdm(list(zip(records, prompts)),
                                         desc=self._progress_desc(1), disable=not show_progress)):
                _results.append(self._postprocess(self._generate(
                    self._build_messages(_record, _prompt, max_new_tokens=max_new_tokens),
                    generation_options), _record, _idx))
            return _results

        nodes = get_numa_node_cpus()
        num_workers = min(self.num_concurrency, len(nodes))
        logger.info("verifier llama_cpp: %d process(es) = min(num_concurrency=%d, NUMA nodes=%d)",
                    num_workers, self.num_concurrency, len(nodes))
        if self.num_concurrency > len(nodes):
            logger.warning("verifier llama_cpp: num_concurrency %d capped to %d "
                           "(single-box CPU inference is memory-bandwidth bound)",
                           self.num_concurrency, num_workers)
        if num_workers <= 1:
            return _sequential()

        model_path = resolve_gguf_path(self.model_name_or_path, self.gguf_filename or _GGUF_FILENAME)
        if model_path is None:
            logger.warning("verifier llama_cpp: multiprocessing needs a local GGUF file, but "
                           "model_name_or_path is a hub repo id -> running single process")
            return _sequential()

        _cpu_budget = get_available_cpu_count()
        num_threads_per_worker = max(1, _cpu_budget // num_workers)
        logger.info("verifier llama_cpp: %d threads/process = cpu budget %d / %d processes "
                    "(NUMA-pinned, %d records)", num_threads_per_worker, _cpu_budget, num_workers, len(records))
        payloads = [
            {"messages": self._to_llama_chat(self._build_messages(_record, _prompt, max_new_tokens=max_new_tokens)),
             **generation_options}
            for _record, _prompt in zip(records, prompts)
        ]
        sizes = split_resources(len(records), num_workers)
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        procs, _start = [], 0
        for _widx, _size in enumerate(sizes):
            _idxs = list(range(_start, _start + _size))
            _start += _size
            if not _idxs:
                continue
            _proc = ctx.Process(
                target=_llama_cpp_shard_worker,
                args=(model_path, self._resolve_num_context_tokens(), num_threads_per_worker,
                      sorted(nodes[_widx % len(nodes)]),
                      _idxs, [payloads[_i] for _i in _idxs], result_queue),
            )
            _proc.start()
            procs.append(_proc)
        # Drain results as workers stream them (one (index, text) per sample, then a DONE sentinel
        # per worker). Poll with a timeout so a worker that dies mid-shard (OOM / load failure)
        # can't hang the parent: once every worker has exited, stop waiting and leave its remaining
        # records as None.
        results: List[Optional[Dict[str, Any]]] = [None] * len(records)
        finished_workers = 0
        with tqdm(total=len(records), desc=self._progress_desc(num_workers),
                  disable=not show_progress) as _pbar:
            while finished_workers < len(procs):
                try:
                    _item = result_queue.get(timeout=5.0)
                except queue.Empty:
                    if all(_proc.exitcode is not None for _proc in procs):
                        logger.warning("verifier llama_cpp: %d/%d workers signalled done before all "
                                       "exited; some exited without finishing their shard",
                                       finished_workers, len(procs))
                        break
                    continue
                if _item == _LLAMA_CPP_WORKER_DONE:
                    finished_workers += 1
                    continue
                _idx, _text = _item
                results[_idx] = self._postprocess(_text, records[_idx], _idx)
                _pbar.update(1)
        for _proc in procs:
            _proc.join()
        # records from a worker that died before emitting them stay None -> null result
        return [_r if _r is not None else self._postprocess(None) for _r in results]

    def _judge_api(self, records, prompts, generation_options, show_progress=True) -> List[Dict[str, Any]]:
        """API backend with ``num_concurrency`` in-flight requests via an asyncio Semaphore.
        The progress bar advances as each request completes (order-preserving gather)."""
        async def _run():
            semaphore = asyncio.Semaphore(self.num_concurrency)
            _pbar = tqdm(total=len(records), desc=self._progress_desc(self.num_concurrency),
                         disable=not show_progress)
            async def _one(_record, _message, _idx):
                _response = await chat_completion_async(
                    api_name=self.api_name,
                    messages=self._build_messages(_record, _message),
                    generation_options=generation_options,
                    semaphore=semaphore,
                    timeout=self.timeout, max_retry=self.max_retry,
                    wait_between_retry=self.wait_between_retry,
                )
                _result = self._postprocess(_response, _record, _idx)
                _pbar.update(1)
                return _result
            try:
                return await asyncio.gather(*[_one(_r, _p, _i)
                                              for _i, (_r, _p) in enumerate(zip(records, prompts))])
            finally:
                _pbar.close()
        return asyncio.run(_run())

    def evaluate(
        self,
        records: List[Union[Dict[str, Any], Record]],
        judge_prompt_list: Optional[Union[str, List[str]]] = None,
        target_metrics: Optional[List[str]] = None,
        exclude_rubrics: Optional[List[str]] = None,
        generation_options: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Verify every record, collecting under each metric name (default
        ``verifier_score``). Same result shape as :meth:`JudgeEvaluator.evaluate`."""
        if not target_metrics:
            target_metrics = ["verifier_score"]
        if isinstance(judge_prompt_list, str):
            judge_prompt_list = [judge_prompt_list] * len(records)
        prompts = [
            (judge_prompt_list[_i] if (judge_prompt_list and _i < len(judge_prompt_list)) else None)
            for _i in range(len(records))
        ]
        _generation_options = self._build_generation_options(generation_options)

        # Dispatch by engine — one _judge_<engine> helper each (their docstrings hold the
        # per-engine parallelism model).
        if self.engine == InferenceEngine.huggingface:
            judge_results = self._judge_huggingface(records, prompts, _generation_options, show_progress)
        elif self.engine == InferenceEngine.llama_cpp:
            judge_results = self._judge_llama_cpp(records, prompts, _generation_options, show_progress)
        elif self.engine in _API_ENGINES:
            judge_results = self._judge_api(records, prompts, _generation_options, show_progress)
        else:
            raise ValueError(f"unsupported verifier engine: {self.engine}")

        metrics: Dict[str, List] = defaultdict(list)
        group_metrics: Dict[str, Any] = dict()
        sample_metrics: List[Dict[str, Any]] = [dict() for _ in records]
        responses: List[Dict[str, Any]] = [dict() for _ in records]
        _alias = self._verifier_alias()
        for _target_metric in target_metrics:
            metrics, group_metrics, sample_metrics = self._collect_judge_results(
                target_metric=_target_metric,
                metrics=metrics, group_metrics=group_metrics, sample_metrics=sample_metrics,
                records=records,
                # rename the numeric ``accuracy`` field to the alias so _collect surfaces the
                # score as ``verifier_score/{alias}`` (distinguishes verifiers / checkpoints, and
                # lets several coexist in one metrics dict). _collect also pops keys, so these
                # throwaway copies double as the pop buffer.
                judge_results=[
                    {(_alias if _k == "accuracy" else _k): _v for _k, _v in r.items()}
                    for r in judge_results
                ],
                exclude_rubrics=exclude_rubrics,
            )
            for _idx, _result in enumerate(judge_results):
                responses[_idx][_target_metric] = _result.get("response", None)

        metrics = {_name: np.nanmean(_vals) for _name, _vals in metrics.items()}
        for _group_name, _group in group_metrics.items():
            _num = max((len(_v) for _v in _group.values() if isinstance(_v, list)), default=0)
            for _name, _vals in _group.items():
                group_metrics[_group_name][_name] = np.mean(_vals)
            group_metrics[_group_name] = dict(group_metrics[_group_name])
            group_metrics[_group_name]["num_samples"] = _num

        return {
            "metrics": metrics,
            "sample_metrics": sample_metrics,
            "group_metrics": group_metrics if group_metrics else None,
            "responses": responses,
        }
