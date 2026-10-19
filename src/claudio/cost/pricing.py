"""Claude API prices (config/pricing.toml) and the cost of one call from its `usage`."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from claudio.i18n import tr

PER_TOKEN = 1 / 1_000_000  # prices are per million tokens
CacheTtl = Literal["5m", "1h"]


class PricingError(Exception):
    """Prices file missing or invalid, or a model without a price."""


class ModelPrice(BaseModel):
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(None, ge=0)  # None: input x cache_read_multiplier


class Pricing(BaseModel):
    currency: str = "USD"
    cache_write_5m_multiplier: float = Field(1.25, ge=0)
    cache_write_1h_multiplier: float = Field(2.0, ge=0)
    cache_read_multiplier: float = Field(0.1, ge=0)
    models: dict[str, ModelPrice] = Field(min_length=1)

    def price(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError:
            raise PricingError(
                tr("Nessun prezzo per il modello '{model}' in config/pricing.toml", model=model)
            ) from None

    def cost(self, model: str, usage: Usage) -> float:
        """Cost of one call, in `currency`."""
        p = self.price(model)
        read = p.cache_read if p.cache_read is not None else p.input * self.cache_read_multiplier
        total = (
            usage.input_tokens * p.input
            + usage.output_tokens * p.output
            + usage.cache_read_input_tokens * read
            + usage.cache_write_5m_tokens * p.input * self.cache_write_5m_multiplier
            + usage.cache_write_1h_tokens * p.input * self.cache_write_1h_multiplier
        )
        return total * PER_TOKEN


@dataclass(frozen=True)
class Usage:
    """The token counts of one call. `input_tokens` are the uncached ones, as in the API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0, not {value}")

    @property
    def cache_creation_input_tokens(self) -> int:
        return self.cache_write_5m_tokens + self.cache_write_1h_tokens

    @classmethod
    def from_api(cls, usage: Any, ttl: CacheTtl = "5m") -> Usage:
        """From the API's `response.usage` (object or dict). Its `cache_creation` splits the
        written tokens by TTL; without it they all go to `ttl`, the TTL the request asked for."""
        written = _int(_get(usage, "cache_creation_input_tokens"))
        split = _get(usage, "cache_creation")
        if split is not None:
            w5m = _int(_get(split, "ephemeral_5m_input_tokens"))
            w1h = _int(_get(split, "ephemeral_1h_input_tokens"))
            w5m += max(0, written - w5m - w1h)  # a total larger than its parts: the default TTL
        else:
            w5m, w1h = (written, 0) if ttl == "5m" else (0, written)
        return cls(
            input_tokens=_int(_get(usage, "input_tokens")),
            output_tokens=_int(_get(usage, "output_tokens")),
            cache_read_input_tokens=_int(_get(usage, "cache_read_input_tokens")),
            cache_write_5m_tokens=w5m,
            cache_write_1h_tokens=w1h,
        )


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _int(value: Any) -> int:
    return int(value) if value else 0  # the API sends null for counts it did not use


def load_pricing(path: Path) -> Pricing:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise PricingError(tr("File dei prezzi non trovato: {path}", path=path)) from None
    except tomllib.TOMLDecodeError as exc:
        raise PricingError(tr("{path}: TOML non valido ({error})", path=path, error=exc)) from exc
    try:
        return Pricing.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise PricingError(
            tr("{path}: valori non validi -> {problems}", path=path, problems=problems)
        ) from exc
