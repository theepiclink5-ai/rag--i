"""Load the versioned prompt used for grounded RAG answers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from src.core.settings import resolve_path


class PromptConfigError(ValueError):
    """Raised when a prompt configuration is missing or malformed."""


@dataclass(frozen=True)
class RagAnswerPrompt:
    """Versioned templates shared by MCP evidence and final LLM answering."""

    name: str
    version: str
    system_prompt: str
    user_template: str
    evidence_preamble: str
    insufficient_evidence_template: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RagAnswerPrompt":
        prompt_path = (
            Path(path)
            if path is not None
            else resolve_path("config/prompts/rag_answer.yaml")
        )
        if not prompt_path.is_absolute():
            prompt_path = resolve_path(prompt_path)
        if not prompt_path.exists():
            raise PromptConfigError(f"RAG answer prompt not found: {prompt_path}")

        with prompt_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise PromptConfigError("RAG answer prompt must be a YAML mapping")

        values: Dict[str, str] = {}
        for field_name in (
            "name",
            "version",
            "system_prompt",
            "user_template",
            "evidence_preamble",
            "insufficient_evidence_template",
        ):
            value: Any = data.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise PromptConfigError(
                    f"Missing non-empty prompt field: {field_name}"
                )
            values[field_name] = value.strip()

        if "{query}" not in values["user_template"]:
            raise PromptConfigError("user_template must contain {query}")
        if "{evidence}" not in values["user_template"]:
            raise PromptConfigError("user_template must contain {evidence}")
        if "{query}" not in values["insufficient_evidence_template"]:
            raise PromptConfigError(
                "insufficient_evidence_template must contain {query}"
            )
        return cls(**values)

    def build_user_prompt(self, query: str, evidence: str) -> str:
        """Fill the two supported placeholders without interpreting evidence."""
        return self.user_template.replace("{query}", query).replace(
            "{evidence}", evidence
        )

    def build_insufficient_evidence(self, query: str) -> str:
        return self.insufficient_evidence_template.replace("{query}", query)

    def metadata(self) -> Dict[str, str]:
        return {"prompt_name": self.name, "prompt_version": self.version}
