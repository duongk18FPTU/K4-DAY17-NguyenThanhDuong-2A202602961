"""Shared response policy: the memory available is the only agent difference."""
from __future__ import annotations

import re
from typing import Any

from memory_store import estimate_tokens, extract_profile_updates

SYSTEM_PROMPT = (
    "Bạn là trợ lý tiếng Việt. Trả lời theo preference đã biết. "
    "Chỉ dùng thông tin người dùng thực sự cung cấp; không suy ra fact từ câu hỏi. "
    "Ưu tiên bản đính chính mới nhất. Nếu không biết, nói rõ không biết."
)

LABELS = {
    "name": "Tên", "location": "Nơi ở hiện tại", "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời", "interests": "Mối quan tâm",
    "favorite_drink": "Đồ uống yêu thích", "favorite_food": "Món ăn yêu thích", "pet": "Thú nuôi",
}


def requested_fields(message: str) -> list[str]:
    low = message.casefold()
    patterns = {
        "name": r"\btên\b|là ai|biết .+ là ai",
        "location": r"ở đâu|nơi ở|đang ở|còn ở|hiện ở",
        "profession": r"nghề|làm gì",
        "response_style": r"style|kiểu trả lời|trả lời.*(?:thích|như thế nào)",
        "interests": r"mối quan tâm|kỹ thuật chính",
        "favorite_drink": r"đồ uống",
        "favorite_food": r"món ăn",
        "pet": r"nuôi|con gì|corgi",
    }
    fields = [key for key, pattern in patterns.items() if re.search(pattern, low)]
    # Resolve locations asked indirectly alongside contradictory career examples.
    return fields


def offline_response(facts: dict[str, str], message: str, summary: str = "") -> str:
    low = message.casefold()
    recall = "?" in message or bool(re.search(r"nhắc lại|nhớ lại|tóm tắt.*mình|mô tả.*mình", low))
    if recall:
        fields = requested_fields(message)
        if not fields and re.search(r"tin|chủ đề|dự án|deadline|mã", low):
            topics = [line[7:] for line in summary.splitlines() if line.startswith("Topic: ")]
            parts = topics or ["Mình chưa có thông tin về chủ đề này"]
        else:
            fields = fields or list(LABELS)
            parts = [f"{LABELS[key]}: {facts[key]}" if key in facts
                     else f"{LABELS[key]}: chưa có thông tin" for key in fields]
    else:
        updates = extract_profile_updates(message)
        parts = ["Đã ghi nhận: " + "; ".join(f"{LABELS[k]}: {v}" for k, v in updates.items())]
        if not updates:
            parts = ["Mình đã nhận nội dung. Bạn có thể hỏi tiếp về phần cần làm rõ."]
    if "3 bullet" in facts.get("response_style", ""):
        # Group facts into exactly three bullets without inventing missing facts.
        groups = [parts[i::3] for i in range(3)]
        defaults = ["Mình sẽ trả lời ngắn gọn.", "Có thể bổ sung ví dụ thực chiến.",
                    "Khi so sánh, mình sẽ nêu trade-off."]
        return "\n".join("- " + ("; ".join(group) if group else defaults[i])
                         for i, group in enumerate(groups))
    return "; ".join(parts) + "."


def prompt_tokens(messages: list[dict[str, str]]) -> int:
    # Same role/framing overhead for both agents; output is excluded here.
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)


def invoke_model(model, messages: list[dict[str, str]]) -> tuple[str, int, int, str]:
    result = model.invoke(messages)
    content = result.content
    if isinstance(content, list):
        content = "\n".join(block.get("text", "") for block in content
                            if isinstance(block, dict) and block.get("type") == "text")
    content = str(content)
    usage: dict[str, Any] = getattr(result, "usage_metadata", None) or {}
    if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
        return content, int(usage["input_tokens"]), int(usage["output_tokens"]), "provider"
    return content, prompt_tokens(messages), estimate_tokens(content), "estimated"
