from __future__ import annotations

import json
import unicodedata
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import evaluate_suites, load_conversations, recall_points, run_agent_benchmark
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates, parse_facts
from model_provider import ProviderConfig, build_chat_model, normalize_provider

ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path):
    model = ProviderConfig("openai", "unused-offline-model")
    return LabConfig(ROOT, ROOT / "data", tmp_path / "state", 1000, 4, model, model, offline=True)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    assert store.file_size("an") == 0
    assert store.read_text("an").startswith("# User")
    path = store.write_text("an", "# User\n\nTên: An\nTên: An\n")
    assert path.name == "User.md"
    assert store.file_size("an") == len(path.read_bytes())
    assert store.edit_text("an", "An", "Bình")
    assert "Tên: Bình\nTên: An" in store.read_text("an")
    assert not store.edit_text("an", "missing", "other")
    assert not store.edit_text("an", "", "other")


def test_profile_updates_are_bounded_and_replace_conflicts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    store.upsert_fact("an", "location", "Huế")
    store.upsert_fact("an", "location", "Đà Nẵng")
    size = store.file_size("an")
    for _ in range(100):
        assert not store.upsert_fact("an", "location", "Đà Nẵng")
    assert store.file_size("an") == size
    assert "Huế" not in store.read_text("an")
    with pytest.raises(ValueError):
        store.upsert_fact("an", "arbitrary", "unbounded schema")


def test_profile_path_is_safe_and_ids_do_not_collide(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    ids = ["../../outside", "..\\outside", "CON", "an/b", "an-b", "An", "an"]
    paths = [store.write_text(user, "# User\n") for user in ids]
    assert len(set(paths)) == len(ids)
    assert all(path.resolve().is_relative_to(tmp_path.resolve()) for path in paths)
    with pytest.raises(ValueError):
        store.path_for("")


@pytest.mark.parametrize("message", [
    "Mình tên là Lan?", "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.",
    "Nhắc lại giúp mình: tên, món ăn yêu thích và mình nuôi con gì.",
    "Nếu mình ở Hà Nội thì sao.", "Giả sử mình làm product manager.",
    "Mình đùa là mình làm product manager.", "Mình không còn làm backend engineer.",
    "Bạn mình tên là Lan và mình ở Hà Nội.",
    'Ví dụ: "Mình tên là Lan".', "Hà Nội chỉ là nơi mình bay ra họp hai ngày.",
    "Đừng nói mình ở Huế.", "Có lẽ mình chuyển sang product manager.",
])
def test_extraction_does_not_persist_questions_negations_or_noise(message: str) -> None:
    assert extract_profile_updates(message) == {}


def test_extraction_handles_other_names_and_corrections() -> None:
    assert extract_profile_updates(
        "Mình tên là Minh An. Mình ở Hải Phòng và đang làm data engineer."
    ) == {"name": "Minh An", "location": "Hải Phòng", "profession": "data engineer"}
    assert extract_profile_updates(
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer."
    ) == {"profession": "MLOps engineer"}
    assert extract_profile_updates(
        "À, giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa."
    ) == {"location": "Huế"}


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(150, 2)
    manager.append("long", "user", "Mình tên là Lan. " + "Ngữ cảnh dài. " * 100)
    assert manager.compaction_count("long") == 0  # Tail alone exceeds a soft threshold.
    manager.append("long", "assistant", "Đã nhận.")
    manager.append("long", "user", "Tiếp tục.")
    context = manager.context("long")
    assert context["compactions"] == 1
    assert len(context["messages"]) == 2
    assert parse_facts(context["summary"])["name"] == "Lan"
    assert manager.compaction_count("other") == 0
    context["messages"].clear()
    assert len(manager.context("long")["messages"]) == 2


def test_repeated_compaction_carries_facts_and_applies_correction() -> None:
    manager = CompactMemoryManager(120, 2)
    manager.append("thread", "user", "Mình tên là Lan. Mình ở Huế.")
    for i in range(8):
        manager.append("thread", "user", f"Dự án thứ {i}. " + "Nội dung dài. " * 80)
        manager.append("thread", "assistant", "Đã nhận.")
    manager.append("thread", "user", "Giờ mình đang ở Đà Nẵng chứ không còn ở Huế nữa.")
    for i in range(4):
        manager.append("thread", "user", "Nội dung khác. " * 80)
    facts = parse_facts(manager.context("thread")["summary"])
    assert facts["name"] == "Lan"
    assert facts["location"] == "Đà Nẵng"
    assert manager.compaction_count("thread") > 2
    assert len(manager.context("thread")["summary"]) < 2000


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline, advanced = BaselineAgent(config), AdvancedAgent(config)
    for agent in (baseline, advanced):
        agent.reply("an", "learn", "Mình tên là Minh An. Đồ uống yêu thích là trà xanh.")
        assert "Minh An" in agent.reply("an", "learn", "Mình tên gì?")["response"]
    assert "Minh An" not in baseline.reply("an", "fresh", "Mình tên gì?")["response"]
    # Simulate process restart: only the file profile survives.
    restarted = AdvancedAgent(config)
    answer = restarted.reply("an", "fresh", "Tên mình và đồ uống yêu thích là gì?")["response"]
    assert "Minh An" in answer and "trà xanh" in answer
    assert restarted.compaction_count("learn") == 0


def test_multiple_users_are_isolated(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for agent in (BaselineAgent(config), AdvancedAgent(config)):
        agent.reply("an", "an-thread", "Mình tên là Minh An.")
        assert "Minh An" not in agent.reply("other", "other-thread", "Mình tên gì?")["response"]
        with pytest.raises(ValueError):
            agent.reply("other", "an-thread", "Mình tên gì?")


def test_question_without_question_mark_cannot_poison_profile(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path))
    agent.reply("an", "one", "Đồ uống yêu thích là trà xanh. Mình nuôi một bé mèo tên Mun.")
    before = agent.profile_store.read_text("an")
    agent.reply("an", "two", "Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.")
    agent.reply("an", "three", "Nhắc lại giúp mình: món ăn yêu thích và mình nuôi con gì.")
    assert agent.profile_store.read_text("an") == before


def test_temporary_topic_stays_in_thread_and_survives_compaction(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), compact_threshold_tokens=200, compact_keep_messages=2)
    baseline, advanced = BaselineAgent(config), AdvancedAgent(config)
    for agent in (baseline, advanced):
        agent.reply("an", "work", "Dự án mã ALPHA-7 có deadline thứ Sáu.")
        immediate = agent.reply("an", "work", "Nhắc lại mã dự án và deadline?")["response"]
        assert "ALPHA-7" in immediate and "thứ Sáu" in immediate
        for _ in range(8):
            agent.reply("an", "work", "Thông tin bổ sung. " * 100)
        answer = agent.reply("an", "work", "Nhắc lại mã dự án và deadline?")["response"]
        assert "ALPHA-7" in answer and "thứ Sáu" in answer
        assert "ALPHA-7" not in agent.reply("an", "new-work", "Nhắc lại mã dự án?")["response"]
    assert advanced.compaction_count("work") > 1
    assert "ALPHA-7" not in advanced.profile_store.read_text("an")


def test_stress_corrections_noise_and_style_are_recalled_without_stale_facts(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path))
    data = load_conversations(ROOT / "data" / "advanced_long_context.json")[0]
    for turn in data["turns"]:
        agent.reply(data["user_id"], "learn", turn)
    facts = agent.profile_store.facts(data["user_id"])
    assert facts["name"] == "DũngCT Stress"
    assert facts["profession"] == "MLOps engineer"
    assert facts["location"] == "Đà Nẵng"
    answer = agent.reply(data["user_id"], "probe", data["recall_questions"][0]["question"])["response"]
    assert "Huế" not in answer and "Hà Nội" not in answer and "product manager" not in answer
    assert len(answer.splitlines()) == 3
    assert all(line.startswith("- ") for line in answer.splitlines())
    topic_answer = agent.reply(data["user_id"], "learn", "Nhắc lại các chủ đề tin tức?")["response"]
    assert len(topic_answer.splitlines()) == 3
    assert all(line.startswith("- ") for line in topic_answer.splitlines())


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    data = load_conversations(config.data_dir / "advanced_long_context.json")
    baseline = run_agent_benchmark("Baseline", BaselineAgent(config), data, config)
    advanced = run_agent_benchmark("Advanced", AdvancedAgent(config), data, config)
    uncompressed_config = replace(config, state_dir=tmp_path / "uncompressed", compact_threshold_tokens=10**9)
    uncompressed = run_agent_benchmark("Uncompressed", AdvancedAgent(uncompressed_config), data, uncompressed_config)
    assert advanced.compactions >= 2
    assert advanced.prompt_tokens_processed < baseline.prompt_tokens_processed * 0.75
    assert advanced.prompt_tokens_processed < uncompressed.prompt_tokens_processed * 0.75
    assert advanced.recall_score == uncompressed.recall_score == 1
    assert baseline.recall_score == 0


def test_supplied_datasets_and_benchmark_are_reproducible(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    # Existing profiles must not contaminate the initial benchmark.
    AdvancedAgent(config).profile_store.upsert_fact("dungct", "name", "Wrong User")
    first = evaluate_suites(config)
    assert first == evaluate_suites(config)
    for suite in first["suites"]:
        baseline, advanced = suite["rows"]
        assert baseline["recall_score"] == 0
        assert advanced["recall_score"] == 1
        assert baseline["memory_growth_bytes"] == 0
        assert advanced["memory_growth_bytes"] > 0
    assert AdvancedAgent(config).profile_store.facts("dungct")["name"] == "Wrong User"


def test_metrics_include_recall_and_exclude_output_from_prompt(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = BaselineAgent(config)
    first = agent.reply("an", "one", "Mình tên là An.")
    second = agent.reply("an", "one", "Mình tên gì?")
    assert agent.token_usage("one") == first["agent_tokens"] + second["agent_tokens"]
    assert agent.prompt_token_usage("one") == first["prompt_tokens"] + second["prompt_tokens"]
    assert second["prompt_tokens"] > first["prompt_tokens"]
    assert agent.token_usage("missing") == agent.prompt_token_usage("missing") == 0
    assert estimate_tokens("") == 0 and estimate_tokens("a") == 1
    assert recall_points("PYTHON", ["Python", "AI"]) == 0.5
    assert recall_points(unicodedata.normalize("NFD", "Huế"), ["Huế"]) == 1
    data = [{"id": "tiny", "user_id": "an", "turns": ["Mình tên là An."],
             "recall_questions": [{"question": "Tên mình là gì?", "expected_contains": ["An"]}]}]
    row = run_agent_benchmark("Tiny", agent, data, config)
    assert row.agent_tokens_only == sum(agent.token_usage(t) for t in agent.sessions if t != "one")


@pytest.mark.parametrize("provider,module,class_name", [
    ("openai", "langchain_openai", "ChatOpenAI"),
    ("custom", "langchain_openai", "ChatOpenAI"),
    ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI"),
    ("anthropic", "langchain_anthropic", "ChatAnthropic"),
    ("ollama", "langchain_ollama", "ChatOllama"),
    ("openrouter", "langchain_openrouter", "ChatOpenRouter"),
])
def test_provider_dispatch_without_network(monkeypatch, provider, module, class_name) -> None:
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return "model"
    def fake_import(name):
        assert name == module
        return SimpleNamespace(**{class_name: factory})
    monkeypatch.setattr("model_provider.import_module", fake_import)
    config = ProviderConfig(provider, "test-model", api_key=None if provider == "ollama" else "secret",
                            base_url="http://localhost:9999" if provider in {"ollama", "custom"} else None)
    assert build_chat_model(config) == "model"
    assert calls[0]["model"] == "test-model"
    assert "secret" not in repr(config)
    if config.base_url:
        assert calls[0]["base_url"] == config.base_url


def test_live_path_uses_memory_and_provider_token_metadata(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for agent in (BaselineAgent(config), AdvancedAgent(config)):
        seen = []
        def invoke(messages):
            seen.append(messages)
            return SimpleNamespace(content=[{"type": "text", "text": "Đã nhận."}],
                                   usage_metadata={"input_tokens": 42, "output_tokens": 7})
        agent.langchain_agent = SimpleNamespace(invoke=invoke)
        result = agent.reply("an", "thread", "Mình tên là Minh An.")
        assert result["mode"] == "live" and result["token_source"] == "provider"
        assert agent.token_usage("thread") == 7
        assert agent.prompt_token_usage("thread") == 42
        agent.reply("an", "fresh", "Mình tên gì?")
        assert ("Minh An" in json.dumps(seen[-1], ensure_ascii=False)) == isinstance(agent, AdvancedAgent)


def test_config_loads_environment_and_validates_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthorpic")
    monkeypatch.setenv("LLM_MODEL", "test-main")
    monkeypatch.setenv("JUDGE_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_MODEL", "test-judge")
    monkeypatch.setenv("LAB_STATE_DIR", "state")
    monkeypatch.setenv("LAB_OFFLINE", "true")
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "123")
    monkeypatch.setenv("COMPACT_KEEP_MESSAGES", "2")
    config = load_config(tmp_path)
    assert config.model.provider == "anthropic"
    assert config.model.model_name == "test-main"
    assert config.judge_model.provider == "ollama"
    assert config.judge_model.model_name == "test-judge"
    assert config.state_dir == tmp_path / "state" and config.state_dir.is_dir()
    monkeypatch.setenv("COMPACT_KEEP_MESSAGES", "0")
    with pytest.raises(ValueError):
        load_config(tmp_path)
    with pytest.raises(ValueError):
        normalize_provider("invalid")
    with pytest.raises(ValueError):
        build_chat_model(ProviderConfig("custom", "test"))
