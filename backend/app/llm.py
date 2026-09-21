"""Provider-neutral chat-model factory.

Switching providers is a one-line change in ``.env`` (``DS_LLM_PROVIDER``). All providers
expose the same LangChain ``BaseChatModel`` interface with tool calling, so the agent code
never needs to know which vendor is behind it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from .config import Settings


@dataclass(frozen=True)
class LLMInfo:
    provider: str
    model: str | None
    mode: str  # "llm" | "heuristic"
    detail: str

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "detail": self.detail,
        }


def build_chat_model(settings: Settings) -> tuple[BaseChatModel | None, LLMInfo]:
    provider = settings.resolved_provider()
    model = settings.resolved_model()
    temp = settings.llm_temperature

    if provider == "heuristic":
        return None, LLMInfo(
            provider="heuristic",
            model=None,
            mode="heuristic",
            detail="No LLM API key found — running the deterministic pandas/scikit-learn analyst. Set OPENAI_API_KEY, ANTHROPIC_API_KEY or GROQ_API_KEY (or DS_LLM_PROVIDER=ollama) to enable the LangChain agent.",
        )

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("DS_LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=temp, timeout=60, max_retries=2)
    elif provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("DS_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=model, temperature=temp, timeout=60, max_retries=2, max_tokens=2048
        )
    elif provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("DS_LLM_PROVIDER=groq but GROQ_API_KEY is not set")
        from langchain_groq import ChatGroq

        llm = ChatGroq(model=model, temperature=temp, timeout=60, max_retries=2)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            temperature=temp,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:  # pragma: no cover - guarded by the Settings Literal type
        raise RuntimeError(f"Unknown provider '{provider}'")

    return llm, LLMInfo(
        provider=provider,
        model=model,
        mode="llm",
        detail=f"LangChain tool-calling agent using {provider} / {model}",
    )
