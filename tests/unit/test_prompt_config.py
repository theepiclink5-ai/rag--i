"""Tests for versioned RAG answer prompt configuration."""

from pathlib import Path

import pytest

from src.core.response.prompt_config import PromptConfigError, RagAnswerPrompt


def test_load_default_prompt_and_fill_templates() -> None:
    prompt = RagAnswerPrompt.load()

    assert prompt.name == "rag_answer"
    assert prompt.version == "1.0.0"
    assert "[n]" in prompt.system_prompt
    assert prompt.build_user_prompt("question", "evidence") == (
        "Question:\nquestion\n\nEvidence:\nevidence"
    )
    assert "question" in prompt.build_insufficient_evidence("question")


def test_missing_required_placeholder_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prompt.yaml"
    path.write_text(
        """
name: rag_answer
version: 1.0.0
system_prompt: grounded
user_template: no placeholders
evidence_preamble: cite [n]
insufficient_evidence_template: "No evidence for {query}"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptConfigError, match="user_template must contain"):
        RagAnswerPrompt.load(path)
