"""Provider registry."""

from __future__ import annotations

from app.providers.base import Provider, ProviderRequest, ProviderResult
from app.schemas import ModelConfig


def build_provider(config: ModelConfig) -> Provider:
    if config.provider == "mock":
        from app.providers.mock import MockProvider

        return MockProvider(config.model)
    if config.provider == "openai_compatible":
        from app.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider()
    if config.provider == "anthropic":
        from app.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    raise ValueError(f"Unsupported provider: {config.provider}")


__all__ = ["Provider", "ProviderRequest", "ProviderResult", "build_provider"]
