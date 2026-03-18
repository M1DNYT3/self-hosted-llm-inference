# inference/registry.py
"""LLM backend registry — @register decorator + get_backend factory."""

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from inference.base import BaseLLMBackend

_BACKENDS: dict[str, "type[BaseLLMBackend]"] = {}

_T = TypeVar("_T", bound="BaseLLMBackend")


def register(name: str) -> Callable[[type[_T]], type[_T]]:
    """Class decorator that registers an LLM backend under ``name``.

    Usage::

        @register("remote")
        class VastaiLLMBackend(BaseLLMBackend):
            ...
    """

    def decorator(cls: type[_T]) -> type[_T]:
        _BACKENDS[name] = cls
        return cls

    return decorator


def get_backend(name: str, **kwargs: object) -> "BaseLLMBackend":
    """Instantiate and return a registered backend by name.

    Raises ValueError for unknown backend names.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown LLM backend: '{name}'. Registered: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name](**kwargs)  # type: ignore[arg-type]


def available_backends() -> list[str]:
    """Return the names of all currently registered backends."""
    return sorted(_BACKENDS.keys())
