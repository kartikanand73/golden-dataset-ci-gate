"""Token pricing. All money math uses Decimal."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml

ONE_MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal


def load_pricing(path: str | Path) -> dict[str, ModelPrice]:
    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = data.get("models") or {}
    prices: dict[str, ModelPrice] = {}
    for name, entry in models.items():
        # str() first so YAML floats never leak binary rounding into Decimal.
        prices[str(name)] = ModelPrice(
            input_per_million_usd=Decimal(str(entry["input_per_million_usd"])),
            output_per_million_usd=Decimal(str(entry["output_per_million_usd"])),
        )
    return prices


def estimate_cost(price: ModelPrice | None, input_tokens: int, output_tokens: int) -> Decimal | None:
    """Return the USD cost for one call, or None when the model has no price entry."""
    if price is None:
        return None
    return (
        Decimal(input_tokens) * price.input_per_million_usd
        + Decimal(output_tokens) * price.output_per_million_usd
    ) / ONE_MILLION
