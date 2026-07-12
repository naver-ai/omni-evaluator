"""Reference helpers for converting a raw resource pile into an
omni_evaluator builtin dataset (``data.jsonl`` + ``resources/{images,audios,videos}/``).

This module is *not* a turnkey pipeline. Each benchmark's raw dump has its own
field naming, directory layout, and label conventions, so the actual driver
script is expected to compose these helpers with dataset-specific glue.

Keep this list of helpers in sync with the sections in ``README.md``:
    - build_content()       : one filename/text -> a chat-content dict
    - build_sample()        : one raw item -> one data.jsonl line (messages/label/meta/…)
    - partition_resources() : copy/rename multimodal files into resources/<kind>/
    - write_data_jsonl()    : dump list[dict] -> data.jsonl
    - upload_to_s3()        : mirror the local tree to a S3-compatible bucket
    - build_dataset()       : orchestration example (customize per benchmark)

References
----------
- Path convention:   ``omni-evaluator/datasets/<benchmark>/<split>/``
- Sample schema:     see ``README.md`` §"data.jsonl schema"
- S3 client:         ``omni_evaluator/clients/s3_client.py`` (S3-compatible
  storage checksum quirks handled internally; do NOT re-implement boto3 calls)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1) build_content — one filename/text → a chat-content dict
# ─────────────────────────────────────────────────────────────────────────────
def build_content(
    type_: str,
    value: str,
    **extras: Any,
) -> Dict[str, Any]:
    """Build one ``content`` item for a ``messages[i].content`` list.

    Parameters
    ----------
    type_
        One of ``"text"``, ``"image"``, ``"audio"``, ``"video"``.
    value
        - For ``text``: the natural-language string.
        - For ``image``/``audio``/``video``: the *filename* placed under
          ``resources/<kind>/`` (loader resolves at runtime).
    extras
        Per-type optional metadata carried alongside ``value``:
        - ``image``: e.g. ``ocr`` (list of ``{id, bbox, text, confidence}``)
        - ``video``: e.g. ``duration``, ``fps``, ``width``, ``height``,
          ``codec``, ``subtitle``
        - ``audio``/``text``: usually none

    Returns
    -------
    dict
        ``{"type": ..., "value": ..., **extras}``
    """
    return {"type": type_, "value": value, **extras}


# ─────────────────────────────────────────────────────────────────────────────
# 2) build_sample — one raw item → one data.jsonl line
# ─────────────────────────────────────────────────────────────────────────────
def build_sample(
    raw: Dict[str, Any],
    *,
    index: Union[int, str],
    user_contents: Sequence[Dict[str, Any]],
    label: Union[str, Sequence[str], None] = None,
    options: Optional[Sequence[str]] = None,
    option_contents: Optional[Sequence[str]] = None,
    captions: Optional[Dict[str, Any]] = None,
    questions: Optional[Sequence[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    assistant_contents: Optional[Sequence[Dict[str, Any]]] = None,
    system_contents: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compose one data.jsonl sample matching the builtin loader contract.

    Output shape (all keys present, unused ones may be ``None`` or omitted per
    the benchmark's convention — see existing builtin task datasets for
    concrete precedent):

        {
            "index": <int|str>,
            "messages": [
                {"role": "user", "content": [<content dicts>]},
                # optional: system / assistant turns for multi-turn
            ],
            "label": [<str>, ...],          # answers; list even if single
            "options": [<str>, ...],        # e.g. ["A", "B", "C", "D"] (MC only)
            "option_contents": [<str>, ...],# choice texts (MC only)
            "captions": {"en": [...], ...}, # optional (e.g. ai2d)
            "questions": [...],             # optional multi-question
            "meta": { ... },                # group_metrics group key lives here
        }

    Parameters
    ----------
    raw
        Source item — only used to carry through unused fields; not read here.
    index
        Sample identifier (unique within the dataset split).
    user_contents
        List of content dicts (build via :func:`build_content`) forming
        the user turn's ``content``. Usually one or more multimodal items
        followed by the question text.
    label
        The ground-truth answer(s). Normalized to ``list[str]``.
    options, option_contents
        MC letters (``["A","B","C","D"]``) and their texts. Set both together.
    captions
        Language-keyed caption dict, e.g. ``{"en": [...], "ko": [...]}``.
    questions
        Optional multi-question list.
    meta
        Free-form metadata dict. **The key ``"category"`` is the primary
        grouping axis** used by ``group_metrics`` in downstream evaluation
        (see README §"meta"). Add ``"category"`` for any benchmark you want
        sliced in group metrics.
    assistant_contents, system_contents
        Only for multi-turn / system-prompt-baked samples.

    Returns
    -------
    dict
        One JSON-serializable sample dict.
    """
    messages: List[Dict[str, Any]] = []
    if system_contents:
        messages.append({"role": "system", "content": list(system_contents)})
    messages.append({"role": "user", "content": list(user_contents)})
    if assistant_contents:
        messages.append({"role": "assistant", "content": list(assistant_contents)})

    if label is None:
        _label: List[str] = []
    elif isinstance(label, (list, tuple)):
        _label = [str(v) for v in label]
    else:
        _label = [str(label)]

    sample: Dict[str, Any] = {
        "index": index,
        "messages": messages,
        "label": _label,
    }
    if options is not None:
        sample["options"] = list(options)
    if option_contents is not None:
        sample["option_contents"] = [str(o) for o in option_contents]
    if captions is not None:
        sample["captions"] = captions
    if questions is not None:
        sample["questions"] = list(questions)
    if meta is not None:
        sample["meta"] = meta
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# 3) partition_resources — copy/rename multimodal files into the target tree
# ─────────────────────────────────────────────────────────────────────────────
_MODALITY_SUBDIR = {
    "image": "images",
    "audio": "audios",
    "video": "videos/base",
}


def partition_resources(
    samples: Iterable[Dict[str, Any]],
    *,
    source_root: Union[str, os.PathLike],
    target_root: Union[str, os.PathLike],
    modality_source_subdirs: Optional[Dict[str, str]] = None,
    modality_target_subdirs: Optional[Dict[str, str]] = None,
    copy_mode: str = "hardlink",
) -> Dict[str, int]:
    """Copy referenced multimodal files under ``target_root/resources/<kind>/``.

    Walks ``sample["messages"][*]["content"]`` looking for ``image``/``audio``/
    ``video`` items and materializes each referenced filename into the target
    resources tree.

    Parameters
    ----------
    samples
        Iterable of samples built by :func:`build_sample`.
    source_root
        Directory that contains the *raw* multimodal files. If sample content
        values are already absolute or already prefixed with a modality
        subdir, we resolve accordingly.
    target_root
        Destination benchmark-split directory
        (e.g. ``./out/omni-evaluator/datasets/<bench>/<split>/``).
        ``resources/<subdir>/`` is created under this root.
    modality_source_subdirs
        Optional override, per modality, for where files live inside
        ``source_root`` (e.g. ``{"image": "raw_pngs"}``).
    modality_target_subdirs
        Optional override, per modality, for the subdir under
        ``target_root/resources/`` (defaults: ``images``, ``audios``,
        ``videos/base``). Use this to target a video variant subdir like
        ``{"video": "videos/64_frames"}``.
    copy_mode
        One of ``"hardlink"`` (fast, requires same filesystem), ``"symlink"``,
        or ``"copy"`` (safe, uses ``shutil.copy2``).

    Returns
    -------
    dict
        Per-modality counts of files successfully placed.
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    _t_map = dict(_MODALITY_SUBDIR)
    if modality_target_subdirs:
        _t_map.update(modality_target_subdirs)
    _s_map = modality_source_subdirs or {}

    counts = {"image": 0, "audio": 0, "video": 0}
    for sample in samples:
        for msg in sample.get("messages", []) or []:
            for c in (msg.get("content") or []):
                modality = c.get("type")
                if modality not in counts:
                    continue
                fname = c.get("value")
                if not fname:
                    continue
                src = _resolve_source_path(
                    fname=fname,
                    source_root=source_root,
                    source_subdir=_s_map.get(modality),
                )
                if src is None or not src.exists():
                    logger.warning("missing %s source for %r", modality, fname)
                    continue
                dst_dir = target_root / "resources" / _t_map[modality]
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / Path(fname).name
                _place(src, dst, mode=copy_mode)
                counts[modality] += 1
    return counts


def _resolve_source_path(
    fname: str,
    source_root: Path,
    source_subdir: Optional[str],
) -> Optional[Path]:
    # Absolute path stored in sample → trust as-is.
    p = Path(fname)
    if p.is_absolute():
        return p
    # Prefer explicit source_subdir override, then bare source_root.
    for candidate in (source_root / (source_subdir or "") / fname, source_root / fname):
        if candidate.exists():
            return candidate
    return None


def _place(src: Path, dst: Path, *, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # cross-device / permission → fall back to copy
            pass
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# 4) write_data_jsonl — dump samples to data.jsonl
# ─────────────────────────────────────────────────────────────────────────────
def write_data_jsonl(
    samples: Iterable[Dict[str, Any]],
    target_root: Union[str, os.PathLike],
    *,
    filename: str = "data.jsonl",
) -> Path:
    """Write samples to ``<target_root>/data.jsonl`` (one JSON per line)."""
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    out = target_root / filename
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5) upload_to_s3 — mirror the built tree to a S3-compatible bucket
# ─────────────────────────────────────────────────────────────────────────────
def upload_to_s3(
    local_root: Union[str, os.PathLike],
    remote_prefix: str,
    *,
    bucket_name: str,
    access_key: str,
    secret_key: str,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = "kr-standard",
    num_process: int = 8,
) -> None:
    """Upload ``local_root`` to ``s3://<bucket>/<remote_prefix>/``.

    ``remote_prefix`` should follow the omni_evaluator convention:
    ``omni-evaluator/datasets/<benchmark>/<split>``. Under it, the bucket ends
    up with ``data.jsonl`` and ``resources/{images,audios,videos/<variant>}/``
    — matching what task config.yaml references via ``data_filepath`` and
    ``image_dirpath`` / ``audio_dirpath`` / ``video_dirpath``.

    Uses the repo's ``S3Client`` (S3-compatible storage checksum quirks are
    handled inside — do NOT swap in a bare boto3 call here).
    """
    from omni_evaluator.clients.s3_client import S3Client

    client = S3Client(
        bucket_name=bucket_name,
        access_key=access_key,
        secret_key=secret_key,
        endpoint_url=endpoint_url,
        region=region,
    )
    client.upload_dir(
        dirpath=str(local_root),
        remote_dirpath=remote_prefix,
        num_process=num_process,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6) build_dataset — orchestration example (customize per benchmark)
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(
    raw_items: Iterable[Dict[str, Any]],
    *,
    benchmark: str,
    split: str,
    source_root: Union[str, os.PathLike],
    output_root: Union[str, os.PathLike],
    sample_builder: Callable[[Dict[str, Any], int], Dict[str, Any]],
    modality_source_subdirs: Optional[Dict[str, str]] = None,
    modality_target_subdirs: Optional[Dict[str, str]] = None,
    copy_mode: str = "hardlink",
    s3_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Dict[str, int]]:
    """End-to-end example that ties the helpers together for one split.

    Typical caller code:

        def _sample(raw, idx):
            contents = [
                build_content("image", raw["image_filename"]),
                build_content("text",  raw["question"]),
            ]
            return build_sample(
                raw, index=idx,
                user_contents=contents,
                label=raw["gt"],
                options=list("ABCD"), option_contents=raw["choices"],
                meta={"category": raw["subject"]},
            )

        build_dataset(
            raw_items=iter_jsonl("/data/raw/foo/all.jsonl"),
            benchmark="foo", split="test",
            source_root="/data/raw/foo/images",
            output_root="./out",
            sample_builder=_sample,
            s3_kwargs={"bucket_name": ..., "access_key": ..., ...},
        )
    """
    target_root = Path(output_root) / "omni-evaluator" / "datasets" / benchmark / split
    target_root.mkdir(parents=True, exist_ok=True)

    samples = [sample_builder(raw, idx) for idx, raw in enumerate(raw_items)]
    counts = partition_resources(
        samples,
        source_root=source_root,
        target_root=target_root,
        modality_source_subdirs=modality_source_subdirs,
        modality_target_subdirs=modality_target_subdirs,
        copy_mode=copy_mode,
    )
    out = write_data_jsonl(samples, target_root)
    logger.info("wrote %d samples → %s (resources: %s)", len(samples), out, counts)

    if s3_kwargs:
        remote_prefix = f"omni-evaluator/datasets/{benchmark}/{split}"
        upload_to_s3(target_root, remote_prefix, **s3_kwargs)
        logger.info("uploaded → s3://%s/%s", s3_kwargs["bucket_name"], remote_prefix)

    return out, counts
