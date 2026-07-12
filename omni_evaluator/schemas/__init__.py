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

from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from typing import Any, Dict


@dataclass
class SchemaInterface:
    def to_dict(self) -> Dict[str, Any]:
        """Convert this dataclass to a plain dict, recursively calling to_dict() on nested dataclasses.

        ``_``-prefixed instance attributes (e.g. ``_raw_yaml`` attached by the
        yaml-driven ``from_engine`` path) are treated as transient/private
        and excluded — they are persisted separately by the caller when
        needed (see ``_output["yaml"]`` in infer.py/evaluate.py).
        """
        output = asdict(self)
        for k, v in self.__dict__.items():
            if k in output: continue
            if k.startswith("_"): continue          # transient private attr
            if is_dataclass(v):
                v = v.to_dict()
            output[k] = v
        return output

    def __getitem__(self, key: str) -> Any:
        """Return the field value for *key*; raise KeyError if *key* is not a dataclass field."""
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(f'key is not a valid field: {key}')

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the field *key* to *value*; raise KeyError if *key* is not a dataclass field."""
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            raise KeyError(f'key is not a valid field: {key}')

    def __contains__(self, key: str) -> bool:
        """Return True if *key* is a declared dataclass field."""
        return key in self.__dataclass_fields__

    def get(self, key: str, default: Any = None) -> Any:
        """dict-like ``.get(key, default)``.

        Returns the field value when *key* is a declared dataclass field;
        otherwise *default*. Mirrors ``dict.get`` so call-sites that don't know
        whether the underlying object is still a dict (e.g. cached snapshot)
        or already a hydrated SchemaInterface instance (post ``TaskConfig.ensure``)
        keep working without TypeErrors.

        Note: like ``__getitem__`` this is field-name based; it does not fall
        back to arbitrary instance attributes set outside ``@dataclass``.
        """
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return default

    def pop(self, key: str, *args) -> Any:
        """Return the value for *key* and reset it to its default (or None); raise KeyError if absent and no default given."""
        if key not in self.__dataclass_fields__:
            if args:
                return args[0]
            raise KeyError(f'key is not a valid field: {key}')
        value = getattr(self, key)
        f = self.__dataclass_fields__[key]
        if f.default is not MISSING:
            setattr(self, key, f.default)
        elif f.default_factory is not MISSING:
            setattr(self, key, f.default_factory())
        else:
            setattr(self, key, None)
        return value

    @classmethod
    def from_kwargs(cls, **kwargs) -> "SchemaInterface":
        """Construct an instance from *kwargs*, silently ignoring keys that are not dataclass fields."""
        field_names = {field.name for field in fields(cls)}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in field_names}
        return cls(**filtered_kwargs)

    @classmethod
    def merge(cls, base: Any, overlay: Any, *, raw_yaml: Any = None) -> Any:
        """Overlay *overlay* onto *base*, recursively.

        Use case: a yaml-rebuilt config (``overlay``) needs to be applied on
        top of a cached snapshot (``base``) so that yaml-authored edits take
        effect while runtime-populated fields the yaml doesn't know about
        (e.g. ``num_records`` set by ``infer.py``) survive.

        Two policies, selected by whether the source yaml's raw dict is known
        for *overlay*:

        1. **yaml-aware** — ``raw_yaml`` is provided (explicit param) or
           ``overlay._raw_yaml`` is set (attached by ``from_engine`` for
           yaml-driven engines like builtin). A field is treated as "authored"
           iff its key appears in the raw yaml dict. Authored fields overlay
           the base regardless of value — including an explicit ``null``.
           Non-authored fields are skipped, so the base survives::

                yaml has "field: null"  →  overlay.field = None  →  base.field = None
                yaml omits "field"      →  overlay.field = None  →  base.field unchanged

        2. **non-None fallback** — no raw yaml available (other engines,
           hydrated-from-dict bases). ``None`` is treated as "not authored"
           so the base survives; any other value overlays::

                base   overlay  result
                ─────────────────────────────────────
                v      None     v       # treated as not authored -> keep base
                v      w        w       # overlay authored -> overlay
                v      v        v       # idempotent
                None   w        w
                None   None     None

        Nested dataclass fields recurse so a partial yaml override at a leaf
        doesn't blow away the cached subtree. For yaml-aware recursion the
        corresponding sub-dict of ``raw_yaml`` is sliced down per field.

        Args:
            base:    cached value (dataclass instance, raw dict, or None).
                     Raw dict is hydrated to *overlay*'s class first (uses
                     ``from_dict`` if defined, else field-filtered ``__init__``).
            overlay: rebuilt value (dataclass instance or None).
            raw_yaml: raw yaml dict that produced *overlay* (or its subtree,
                     during recursion). When ``None``, falls back to
                     ``overlay._raw_yaml`` for the top-level call.

        Returns:
            merged instance (typically of ``type(overlay)``). Defers to
            *overlay* when types can't be merged.

        Known limits (intentional trade-offs, fallback policy only):
          - Container defaults (``[]`` / ``{}`` / ``0`` / ``False``) are
            non-None, so they DO overlay — be aware when adding fields with
            those defaults that are also runtime-populated. (yaml-aware mode
            handles this correctly by consulting the raw yaml.)
        """
        import copy as _copy
        if overlay is None:
            return base
        if base is None:
            return overlay
        # raw dict base → hydrate to overlay's class for uniform field access
        if isinstance(base, dict):
            _ovr_cls = type(overlay) if is_dataclass(overlay) else cls
            if hasattr(_ovr_cls, "from_dict"):
                base = _ovr_cls.from_dict(base)
            else:
                _valid = {f.name for f in fields(_ovr_cls)}
                base = _ovr_cls(**{k: v for k, v in base.items() if k in _valid})
        if not (is_dataclass(base) and is_dataclass(overlay)):
            return overlay
        # yaml-aware mode is opt-in: explicit raw_yaml kwarg, or the overlay
        # carries one attached by its from_engine path. None at top-level →
        # fallback (non-None overlay) policy.
        if raw_yaml is None:
            raw_yaml = getattr(overlay, "_raw_yaml", None)
        merged = _copy.deepcopy(base)
        for f in fields(overlay):
            new_val = getattr(overlay, f.name)
            if isinstance(raw_yaml, dict):
                # yaml-aware: authored ⇔ key present in raw yaml
                if f.name not in raw_yaml:
                    continue                            # not authored — keep base
                _sub_raw = raw_yaml.get(f.name)
                _sub_raw = _sub_raw if isinstance(_sub_raw, dict) else None
            else:
                # fallback: None means "not authored"
                if new_val is None:
                    continue
                _sub_raw = None
            old_val = getattr(merged, f.name, None)
            if is_dataclass(new_val) and is_dataclass(old_val):
                # recurse so partial nested overrides preserve cached leaves
                setattr(merged, f.name, type(new_val).merge(old_val, new_val, raw_yaml=_sub_raw))
            else:
                setattr(merged, f.name, new_val)
        # Carry the overlay's raw yaml onto the merged result so the next
        # save path persists the current-run yaml (history + input for the
        # following resume's yaml-aware merge).
        if isinstance(raw_yaml, dict):
            merged._raw_yaml = raw_yaml
        return merged