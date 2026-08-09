"""Dead-simple JSON-file memoization for cookbook LLM calls."""

from __future__ import annotations

import functools
import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path

MAX_KEY_CHARS = (
    200  # keys longer than this (e.g. whole documents) are stored as a sha256 digest
)


class JsonCache:
    """Memoize a function's JSON-serializable results to a single JSON file.

    Keys include the qualified function name, public bound-instance state, and ``|``-joined
    arguments, so configured clients and functions can share one file. Keys longer than
    ``MAX_KEY_CHARS`` are replaced by their sha256 digest.

    >>> cache = JsonCache(Path(__file__).with_name("json_cache.json"))
    >>> @cache
    ... def ask(model: str, temperature: float | None, k: int): ...
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound_instance = getattr(fn, "__self__", None)
            public_state = (
                {
                    name: value
                    for name, value in vars(bound_instance).items()
                    if not name.startswith("_")
                }
                if bound_instance is not None
                else {}
            )
            key = "|".join(
                [f"{fn.__module__}.{fn.__qualname__}", str(public_state)]
                + [str(a) for a in args]
                + [f"{k}={v}" for k, v in sorted(kwargs.items())]
            )
            if len(key) > MAX_KEY_CHARS:
                key = hashlib.sha256(key.encode()).hexdigest()
            if key not in self._data:
                result = fn(*args, **kwargs)
                model_dump = getattr(result, "model_dump", None)
                if callable(model_dump):
                    result = model_dump(mode="json")
                with self._lock:
                    self._data[key] = result
                    self.path.write_text(
                        json.dumps(self._data, indent=2, sort_keys=True)
                    )
            # deep-copy via JSON so callers can mutate results without corrupting the cache
            return json.loads(json.dumps(self._data[key]))

        return wrapper
