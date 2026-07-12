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

from enum import Enum


class TaskType(str, Enum):
    """What capability the model is exercising (task-level, not output format)."""
    # Vision-Language
    visual_question_answering: str = "visual_question_answering"
    figure_question_answering: str = "figure_question_answering"        # MathVista FQA
    geometry_problem_solving: str = "geometry_problem_solving"          # MathVista GPS
    math_word_problem: str = "math_word_problem"                        # MathVista MWP
    textbook_question_answering: str = "textbook_question_answering"    # MathVista TQA
    image_captioning: str = "image_captioning"
    image_generation: str = "image_generation"
    image_grounding: str = "image_grounding"          # referring expression / phrase grounding (RefCOCO etc.)
    # Audio / Speech
    speech_recognition: str = "speech_recognition"   # ASR / transcription
    speech_translation: str = "speech_translation"
    audio_understanding: str = "audio_understanding"  # scene QA, emotion, music, …
    audio_captioning: str = "audio_captioning"
    # Language / Code / Agent
    agent: str = "agent"
    code_generation: str = "code_generation"
    tool_calling: str = "tool_calling"
    # Video
    video_question_answering: str = "video_question_answering"
    video_captioning: str = "video_captioning"
    video_grounding: str = "video_grounding"          # spatial grounding in video
    temporal_grounding: str = "temporal_grounding"    # localize moments by language (e.g. Charades-STA)
    # Fallback
    unknown: str = "unknown"


class SubtaskType(str, Enum):
    """Domain or output-format dimension within a task (orthogonal to TaskType)."""
    # Input domain
    document: str = "document"         # DocVQA, TextVQA, OCR, InfoVQA
    diagram: str = "diagram"           # AI2D, scientific diagrams
    chart: str = "chart"               # ChartQA (subset of document)
    table: str = "table"               # table
    web_ui: str = "web_ui"             # web_ui
    scene: str = "scene"               # natural-image QA
    scene_text: str = "scene_text"     # scene-text VQA (STVQA, TextVQA)
    # Reasoning type
    math: str = "math"                 # MathVista, MathVerse, WeMath
    reasoning: str = "reasoning"       # CharXiv, HallusionBench
    exam: str = "exam"                 # KCSAT, KONET, KGED
    code: str = "code"                 # LiveCodeBench
    hallucination: str = "hallucination"  # POPE, HallusionBench (hallucination eval)
    # Output format
    multiple_choice: str = "multiple_choice"  # MCQ regardless of engine (replaces multichoice)
    open_ended: str = "open_ended"     # short-answer counterpart of MCQ (e.g., KCSAT short-answer natural number)
    freeform: str = "freeform"         # open-ended generation (replaces subtask "generation")
    classification: str = "classification"
    summary: str = "summary"
    # Grounding referent type (RefCOCO testA/testB split semantics)
    people: str = "people"             # people-referring expression (RefCOCO testA)
    object: str = "object"             # object-referring expression (RefCOCO testB)
    # Audio-specific sub-types
    asr: str = "asr"                   # automatic speech recognition
    audio_scene: str = "audio_scene"   # environmental sound understanding
    audio_understanding: str = "audio_understanding"  # general audio QA (e.g., MMSU)
    emotion: str = "emotion"           # emotion / sentiment recognition
    speaker: str = "speaker"           # accent / gender recognition
    music: str = "music"               # music understanding
    # General / catch-all
    general: str = "general"
    # Fallback
    unknown: str = "unknown"


class SpatialGroundingType(str, Enum):
    """Spatial grounding output shape — image / video / STVG.

    Used for any *visual* grounding where the answer is a 2D region:
    referring-expression comprehension (RefCOCO/+/g, Flickr30k-Entities,
    Kosmos), spatial video grounding (per-frame box), and spatio-temporal
    video grounding (STVG, per-frame box tube). The same shape is used both
    for the **input target** (what the metric scores against — e.g.
    ``compute_iou`` expects ``"[x1, y1, x2, y2]"``) and for the **output
    spec** that :class:`SpatialGroundingProcessor` re-emits. Distinct from
    temporal grounding (``TaskType.temporal_grounding``), which is a single
    ``(start_sec, end_sec)`` shape and needs no enum.

    - ``BBOX``  — axis-aligned bounding box (4 nums): ``[x1, y1, x2, y2]``
    - ``POINT`` — single 2D point (2 nums): ``[x, y]``
    - ``QUAD``  — 4-corner polygon (8 nums): ``[x1, y1, x2, y2, x3, y3, x4, y4]``
    """
    BBOX: str = "bbox"
    POINT: str = "point"
    QUAD: str = "quad"
