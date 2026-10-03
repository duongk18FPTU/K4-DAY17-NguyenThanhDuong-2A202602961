from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates, summarize_messages
from model_provider import build_chat_model
from response_engine import SYSTEM_PROMPT, invoke_model, offline_response, prompt_tokens


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Within-thread history only; no profile, summary or cross-thread facts."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.thread_users: dict[str, str] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id in self.thread_users and self.thread_users[thread_id] != user_id:
            raise ValueError("A thread cannot be shared between different users")
        self.thread_users[thread_id] = user_id
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        return self._respond(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        return self._respond(thread_id, message, offline=True)

    def _respond(self, thread_id: str, message: str, offline: bool = False) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        user_message = {"role": "user", "content": message}
        history = session.messages + [user_message]
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        if offline:
            facts = {}
            for item in history:
                if item["role"] == "user":
                    facts.update(extract_profile_updates(item["content"]))
            answer = offline_response(facts, message, summarize_messages(history))
            input_count, output_count, source = prompt_tokens(prompt), estimate_tokens(answer), "estimated"
        else:
            answer, input_count, output_count, source = invoke_model(self.langchain_agent, prompt)
        session.messages.extend([user_message, {"role": "assistant", "content": answer}])
        session.token_usage += output_count
        session.prompt_tokens_processed += input_count
        return {"response": answer, "agent_tokens": output_count, "prompt_tokens": input_count,
                "token_source": source, "mode": "offline" if offline else "live"}

    def _maybe_build_langchain_agent(self):
        # The lab owns thread history explicitly; LangChain provides model transport.
        return None if self.force_offline or self.config.offline else build_chat_model(self.config.model)
