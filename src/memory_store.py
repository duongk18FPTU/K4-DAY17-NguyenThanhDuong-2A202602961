from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_FIELDS = (
    "name", "location", "profession", "response_style", "interests",
    "favorite_drink", "favorite_food", "pet",
)


def estimate_tokens(text: str) -> int:
    """Stable character heuristic, not a tokenizer or an API billing estimate."""
    return math.ceil(len(text.strip()) / 4)


def normalized(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def parse_facts(text: str) -> dict[str, str]:
    facts = {}
    for line in text.splitlines():
        match = re.match(r"^- ([a-z_]+): (.+)$", line)
        if match and match[1] in PROFILE_FIELDS:
            facts[match[1]] = match[2]
    return facts


def render_facts(facts: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {facts[key]}" for key in PROFILE_FIELDS if key in facts)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id.strip():
            raise ValueError("user_id cannot be empty")
        # Hash every ID: unsafe IDs cannot collide with valid IDs after sanitizing.
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", user_id).strip("-")[:64] or "user"
        return Path(self.root_dir) / f"{slug}--{digest}" / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User profile\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so readers never observe half a markdown file.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(normalized(content))
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            return False
        content = self.read_text(user_id)
        if search_text not in content or search_text == replacement:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return parse_facts(self.read_text(user_id))

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        if key not in PROFILE_FIELDS:
            raise ValueError(f"Unknown profile field: {key}")
        value = normalized(" ".join(value.split()))[:512]
        if not value:
            return False
        facts = self.facts(user_id)
        if facts.get(key) == value:
            return False
        # Replace a conflicting field, never append obsolete values.
        facts[key] = value
        self.write_text(user_id, "# User profile\n\n" + render_facts(facts) + "\n")
        return True


def _sentences(text: str):
    for match in re.finditer(r"[^.!?\n]+[.!?]?", normalized(text)):
        sentence = match[0].strip()
        if sentence:
            yield sentence


def _clean(value: str) -> str:
    return value.strip(" \t,;:.!")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative Vietnamese rules for explicit self disclosures.

    Only positive declarations/instructions are accepted. Question, conditional,
    quoted, joke and third-party sentences cannot change persistent facts.
    This is a rule-based gate, not calibrated statistical confidence.
    """
    updates = {}
    proper_name = r"[A-ZĐÀ-Ỹ][\w-]*(?:\s+[A-ZĐÀ-Ỹ][\w-]*){0,5}"
    role = r"(?:[A-Za-z][A-Za-z0-9+./-]*\s+){0,3}(?:engineer|developer|manager|designer|scientist|analyst)|(?:lập trình viên|giáo viên|bác sĩ|kỹ sư)"
    for sentence in _sentences(message):
        low = sentence.casefold()
        if sentence.endswith("?") or re.search(
            r"\b(nếu|giả sử|có lẽ|có thể mình|đùa|câu hỏi|chưa chắc|hay là|nghe nói"
            r"|là gì|tên gì|ở đâu|nuôi con gì|nhắc lại|nhớ lại|bạn có|bạn thử"
            r"|bạn mình|bạn tôi|anh mình|chị mình|bạn nói|trong ví dụ)\b", low
        ) or any(quote in sentence for quote in ('"', "“", "”")):
            continue
        # Process positive clauses independently from a previous negated clause.
        for clause in re.split(r"[,;]|\bchứ\b", sentence):
            c = clause.casefold().strip()
            if re.search(r"\b(không|chưa|đừng)\b", c):
                continue
            name = re.search(r"\b[Mm]ình\s+tên(?:\s+là)?\s+(" + proper_name + r")", clause)
            if name:
                updates["name"] = _clean(name[1])
            location = re.search(
                r"(?:[Mm]ình\s+(?:(?:vẫn|hiện|đang|hiện đang|giờ đang)\s+)?"
                r"(?:(?:sống|sinh sống|làm việc)\s+)?ở|hiện ở|nơi ở hiện tại(?:\s+của mình)?\s+là)"
                r"\s+(" + proper_name + r")", clause
            )
            if location:
                updates["location"] = _clean(location[1])
            job = re.search(
                r"(?:[Mm]ình\s+(?:(?:vẫn|đang|giờ|hiện tại|giờ đang)\s+)?làm"
                r"|đang làm|giờ chuyển sang|[Mm]ình chuyển sang"
                r"|[Nn]ghề nghiệp(?:\s+hiện tại)?(?:\s+của mình)?(?:\s+thì)?(?:\s+vẫn)?\s+là"
                r"|[Nn]ghề)\s+(" + role + r")\b", clause, re.IGNORECASE
            )
            if job:
                updates["profession"] = _clean(job[1])
        for key, prefix in (("favorite_drink", "đồ uống"), ("favorite_food", "món ăn")):
            favorite = re.search(prefix + r"\s+yêu thích(?:\s+của mình)?\s+là\s+([^,;.]+)",
                                 sentence, re.IGNORECASE)
            if favorite and not re.search(r"\b(không|chưa)\b", low[:favorite.end()]):
                updates[key] = _clean(favorite[1])
        pet = re.search(r"[Mm]ình nuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?(\w+)(?:\s+tên\s+(" + proper_name + r"))?", sentence)
        if pet:
            updates["pet"] = pet[1] + (f" tên {pet[2]}" if pet[2] else "")
        if re.search(r"mình (?:vẫn )?(?:thích|đang quan tâm|quan tâm)", low):
            interests = []
            for pattern in (r"\bPython\b", r"\bAI(?: ứng dụng| agent)?\b", r"\bMLOps\b",
                            r"\bRAG\b", r"\bevaluation\b", r"\bJava\b", r"\bRust\b"):
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    interests.append(match[0])
            if interests:
                updates["interests"] = ", ".join(interests)
        # Persist style only when the user explicitly instructs how to reply.
        if re.search(r"mình (?:vẫn )?(?:muốn|thích)|hãy trả lời|style trả lời", low) and re.search(
            r"trả lời|giải thích", low
        ):
            style = []
            if re.search(r"ngắn|gọn", low):
                style.append("ngắn gọn")
            bullet = re.search(r"(\d+)\s+bullet", low)
            if bullet:
                style.append(f"{bullet[1]} bullet")
            elif "bullet" in low:
                style.append("bullet")
            if re.search(r"ví dụ thực (?:tế|chiến)", low):
                style.append("có ví dụ thực chiến" if "thực chiến" in low else "có ví dụ thực tế")
            if "trade-off" in low:
                style.append("ưu tiên so sánh trade-off")
            if style:
                updates["response_style"] = ", ".join(style)
    return updates


def _topic_items(messages: list[dict[str, str]], max_items: int) -> list[str]:
    items = []
    for message in messages:
        if message["role"] != "user":
            continue
        # Keep a short representative sentence rather than entire news articles.
        candidates = [s for s in _sentences(message["content"]) if not s.endswith("?")]
        meaningful = [s for s in candidates if re.search(
            r"tin |chủ đề|góc nhìn|pattern|mục tiêu|quyết định|mã |deadline|dự án",
            s, re.IGNORECASE
        )]
        for sentence in (meaningful or candidates[:1])[:2]:
            snippet = " ".join(sentence.split())[:180]
            if snippet and snippet not in items:
                items.append(snippet)
    return items[-max_items:] if max_items > 0 else []


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    facts = {}
    for message in messages:
        if message["role"] == "user":
            facts.update(extract_profile_updates(message["content"]))
    topics = _topic_items(messages, max_items)
    parts = [render_facts(facts)] if facts else []
    parts.extend(f"Topic: {topic}" for topic in topics)
    return "\n".join(parts)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 1:
            raise ValueError("Invalid compact settings")

    def append(self, thread_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported message role")
        state = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = state["messages"]
        messages.append({"role": role, "content": content})
        load = estimate_tokens(state["summary"]) + sum(estimate_tokens(m["content"]) for m in messages)
        if load <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return
        older, recent = messages[:-self.keep_messages], messages[-self.keep_messages:]
        # Carry previous structured facts forward instead of re-summarizing a
        # truncated summary. Fresh facts replace conflicts in chronological order.
        facts = parse_facts(state["summary"])
        facts.update(parse_facts(summarize_messages(older, max_items=0)))
        topics = [line[7:] for line in state["summary"].splitlines() if line.startswith("Topic: ")]
        topics.extend(_topic_items(older, 6))
        topics = list(dict.fromkeys(topics))[-6:]
        state["summary"] = "\n".join([render_facts(facts)] + [f"Topic: {t}" for t in topics]).strip()
        state["messages"] = recent
        state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        state = self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})
        # Snapshot prevents callers accidentally mutating the manager's history.
        return {"messages": [dict(m) for m in state["messages"]],
                "summary": state["summary"], "compactions": state["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
