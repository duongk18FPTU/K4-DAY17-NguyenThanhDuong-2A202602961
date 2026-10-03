from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    offline: bool = True

    def __post_init__(self) -> None:
        if self.compact_threshold_tokens <= 0 or self.compact_keep_messages < 1:
            raise ValueError("Compact threshold must be positive; keep at least one message")


DEFAULT_MODELS = {
    "openai": "gpt-4o-mini", "custom": "local-model", "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-5", "ollama": "llama3.2", "openrouter": "openai/gpt-4o-mini",
}


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai"))
    key_name = {"gemini": "GEMINI", "custom": "CUSTOM"}.get(provider, provider.upper())
    same_provider = fallback is not None and provider == fallback.provider
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", fallback.model_name if same_provider else DEFAULT_MODELS[provider]),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv(f"{key_name}_API_KEY")
        or (os.getenv("GOOGLE_API_KEY") if provider == "gemini" else None)
        or (fallback.api_key if same_provider else None),
        base_url=os.getenv(f"{prefix}_BASE_URL") or os.getenv(f"{key_name}_BASE_URL")
        or (fallback.base_url if same_provider else None),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
    except ImportError:
        if (root / ".env").exists():
            raise RuntimeError("Install python-dotenv to load .env configuration") from None
    else:
        load_dotenv(root / ".env", override=False)
    state = Path(os.getenv("LAB_STATE_DIR", str(root / "state")))
    if not state.is_absolute():
        state = root / state
    state.mkdir(parents=True, exist_ok=True)
    offline = os.getenv("LAB_OFFLINE", "true").strip().lower()
    if offline not in {"true", "false", "1", "0"}:
        raise ValueError("LAB_OFFLINE must be true/false or 1/0")
    model = _provider_config("LLM")
    return LabConfig(root, root / "data", state.resolve(),
                     int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000")),
                     int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
                     model, _provider_config("JUDGE", model), offline in {"true", "1"})
