from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import normalized


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    conversations = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(conversations, list):
        raise ValueError("Dataset must be a list of conversations")
    for conversation in conversations:
        for key in ("id", "user_id", "turns", "recall_questions"):
            if key not in conversation:
                raise ValueError(f"Missing dataset field: {key}")
        if not all(isinstance(turn, str) for turn in conversation["turns"]):
            raise ValueError("Conversation turns must be strings")
        for question in conversation["recall_questions"]:
            if not isinstance(question["question"], str) or not isinstance(question["expected_contains"], list):
                raise ValueError("Invalid recall question")
    return conversations


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of case-insensitive expected substrings, including partial recall."""
    if not expected:
        return 1.0
    answer = normalized(answer).casefold()
    return sum(normalized(fact).casefold() in answer for fact in expected) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Recall plus a small brevity score; not a semantic quality judge."""
    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    if recall == 0:
        return 0.0
    brevity = min(1.0, 120 / max(1, len(answer.split())))
    return 0.85 * recall + 0.15 * brevity


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config: LabConfig,
    details: list[dict[str, Any]] | None = None,
) -> BenchmarkRow:
    users = {c["user_id"] for c in conversations}
    memory_size = getattr(agent, "memory_file_size", lambda user_id: 0)
    initial_size = sum(memory_size(user) for user in users)
    threads = set()
    recall_scores, quality_scores = [], []
    for index, conversation in enumerate(conversations):
        user = conversation["user_id"]
        thread = f"{agent_name}:conversation:{index}:{conversation['id']}"
        threads.add(thread)
        for turn in conversation["turns"]:
            agent.reply(user, thread, turn)
        # Probe each question in an independent fresh thread: no answer leakage
        # between recall questions, and no access to expected answers by agents.
        for question_index, question in enumerate(conversation["recall_questions"]):
            recall_thread = f"{agent_name}:recall:{index}:{question_index}"
            threads.add(recall_thread)
            result = agent.reply(user, recall_thread, question["question"])
            answer = result["response"]
            score = recall_points(answer, question["expected_contains"])
            quality = heuristic_quality(answer, question["expected_contains"])
            recall_scores.append(score)
            quality_scores.append(quality)
            if details is not None:
                details.append({"agent": agent_name, "conversation": conversation["id"],
                                "question": question["question"], "answer": answer,
                                "expected_contains": question["expected_contains"], "recall": score,
                                "token_source": result["token_source"]})
    return BenchmarkRow(
        agent_name,
        sum(agent.token_usage(thread) for thread in threads),
        sum(agent.prompt_token_usage(thread) for thread in threads),
        sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        sum(memory_size(user) for user in users) - initial_size,
        sum(agent.compaction_count(thread) for thread in threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
               "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
                 f"{row.recall_score:.1%}", f"{row.response_quality:.3f}",
                 str(row.memory_growth_bytes), str(row.compactions)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def evaluate_suites(config: LabConfig, live: bool = False, ablation: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": "live" if live else "offline",
        "metrics": "Offline tokens are ceil(chars/4) estimates. Quality is a recall/brevity heuristic.",
        "compact_threshold_tokens": config.compact_threshold_tokens,
        "compact_keep_messages": config.compact_keep_messages,
        "suites": [],
    }
    for title, filename in (("Standard Benchmark", "conversations.json"),
                            ("Long-Context Stress Benchmark", "advanced_long_context.json")):
        path = config.data_dir / filename
        conversations = load_conversations(path)
        details: list[dict[str, Any]] = []
        rows = []
        # A fresh directory per agent and suite keeps reruns independent of
        # existing personal profiles. No preexisting user state is removed.
        variants = ["Baseline", "Advanced"]
        if ablation and filename == "advanced_long_context.json":
            variants.append("Advanced (no compact)")
        for variant in variants:
            with TemporaryDirectory(prefix="memory-lab-") as temporary:
                isolated = replace(config, state_dir=Path(temporary), offline=not live)
                if variant == "Advanced (no compact)":
                    isolated = replace(isolated, compact_threshold_tokens=10**9)
                agent = (BaselineAgent(isolated, force_offline=not live) if variant == "Baseline"
                         else AdvancedAgent(isolated, force_offline=not live))
                rows.append(run_agent_benchmark(variant, agent, conversations, isolated, details))
        report["suites"].append({
            "title": title, "dataset": filename,
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "conversations": len(conversations),
            "turns": sum(len(c["turns"]) for c in conversations),
            "recall_questions": sum(len(c["recall_questions"]) for c in conversations),
            "rows": [asdict(row) for row in rows], "recall_details": details,
        })
    return report


def report_markdown(report: dict[str, Any]) -> str:
    lines = ["# Memory Systems Benchmark", "", f"Mode: {report['mode']}", "",
             report["metrics"], "",
             f"Compact threshold: {report['compact_threshold_tokens']}; "
             f"recent messages kept: {report['compact_keep_messages']}.",
             "Counts include conversation turns and fresh-thread recall probes.", ""]
    for suite in report["suites"]:
        lines.extend([f"## {suite['title']}", "",
                      f"{suite['conversations']} conversations, {suite['turns']} turns, "
                      f"{suite['recall_questions']} recall questions.", "",
                      format_rows([BenchmarkRow(**row) for row in suite["rows"]]), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare agent memory on both Vietnamese datasets")
    parser.add_argument("--live", action="store_true", help="Explicitly enable paid/provider model calls")
    parser.add_argument("--ablation", action="store_true", help="Also measure Advanced without compaction")
    parser.add_argument("--output", type=Path, help="Write benchmark markdown")
    parser.add_argument("--json", type=Path, help="Write metrics and individual recall answers")
    args = parser.parse_args()
    config = load_config(Path(__file__).resolve().parent.parent)
    report = evaluate_suites(config, live=args.live, ablation=args.ablation)
    markdown = report_markdown(report)
    print(markdown)
    for path, content in ((args.output, markdown),
                          (args.json, json.dumps(report, ensure_ascii=False, indent=2))):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
