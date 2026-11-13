"""Configuration loading and validation for the Modular RAG MCP Server."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
# Anchored to this file's location: <repo>/src/core/settings.py → parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Default absolute path to settings.yaml
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """Resolve a repo-relative path to an absolute path.

    If *relative* is already absolute it is returned as-is.  Otherwise
    it is resolved against :data:`REPO_ROOT`.

    >>> resolve_path("config/settings.yaml")  # doctest: +SKIP
    PosixPath('/home/user/Modular-RAG-MCP-Server/config/settings.yaml')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """Raised when settings validation fails."""


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_dotenv_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file without overriding env vars."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _expand_env_placeholders(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} placeholders in loaded YAML values."""
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    full_match = _ENV_PATTERN.fullmatch(value.strip())
    if full_match:
        return os.environ.get(full_match.group(1))

    return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


def _optional_int(data: Dict[str, Any], key: str, path: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _optional_number(
    data: Dict[str, Any], key: str, path: str, default: float
) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _optional_bool(data: Dict[str, Any], key: str, path: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _optional_str(data: Dict[str, Any], key: str, path: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    # Azure/OpenAI-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    # Azure-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int
    candidate_top_k: int = 20
    timeout: float = 15.0
    batch_size: int = 16
    max_length: int = 512
    fallback_on_error: bool = True
    cache_dir: str = "./data/models/huggingface/hub"
    execution_mode: str = "in_process"
    startup_timeout: float = 60.0
    local_files_only: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 2
    retry_base_delay: float = 0.5


@dataclass(frozen=True)
class CitationGuardSettings:
    enabled: bool = True
    min_text_length: int = 30
    strong_score_threshold: float = 0.60
    weak_score_threshold: float = 0.20
    min_strong_evidence: int = 1


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: List[str]


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class VisionLLMSettings:
    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int
    chunk_refiner: Optional[Dict[str, Any]] = None  # 动态配置
    metadata_enricher: Optional[Dict[str, Any]] = None  # 动态配置


@dataclass(frozen=True)
class MinerUPdfParserSettings:
    api_key: Optional[str] = None
    model_version: str = "vlm"
    language: str = "ch"
    is_ocr: bool = True
    enable_table: bool = True
    enable_formula: bool = True
    timeout_seconds: int = 600
    poll_interval_seconds: int = 5
    save_raw_result: bool = True


@dataclass(frozen=True)
class PdfParserSettings:
    primary: str
    fallback_enabled: bool
    fallback_provider: str
    fallback_threshold: float
    mineru: Optional[MinerUPdfParserSettings] = None


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: Optional[IngestionSettings] = None
    vision_llm: Optional[VisionLLMSettings] = None
    pdf_parser: Optional[PdfParserSettings] = None
    citation_guard: Optional[CitationGuardSettings] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")

        citation_guard_settings = None
        if "citation_guard" in data:
            citation_guard = _require_mapping(data, "citation_guard", "settings")
            citation_guard_settings = CitationGuardSettings(
                enabled=_optional_bool(
                    citation_guard, "enabled", "citation_guard", True
                ),
                min_text_length=_optional_int(
                    citation_guard, "min_text_length", "citation_guard", 30
                ),
                strong_score_threshold=_optional_number(
                    citation_guard,
                    "strong_score_threshold",
                    "citation_guard",
                    0.60,
                ),
                weak_score_threshold=_optional_number(
                    citation_guard,
                    "weak_score_threshold",
                    "citation_guard",
                    0.20,
                ),
                min_strong_evidence=_optional_int(
                    citation_guard,
                    "min_strong_evidence",
                    "citation_guard",
                    1,
                ),
            )

        ingestion_settings = None
        if "ingestion" in data:
            ingestion = _require_mapping(data, "ingestion", "settings")
            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                chunk_refiner=ingestion.get("chunk_refiner"),  # 可选配置
                metadata_enricher=ingestion.get("metadata_enricher"),  # 可选配置
            )

        pdf_parser_settings = None
        if "pdf_parser" in data:
            pdf_parser = _require_mapping(data, "pdf_parser", "settings")
            mineru_settings = None
            if "mineru" in pdf_parser:
                mineru = _require_mapping(pdf_parser, "mineru", "pdf_parser")
                mineru_settings = MinerUPdfParserSettings(
                    api_key=mineru.get("api_key"),
                    model_version=str(mineru.get("model_version", "vlm")),
                    language=str(mineru.get("language", "ch")),
                    is_ocr=bool(mineru.get("is_ocr", True)),
                    enable_table=bool(mineru.get("enable_table", True)),
                    enable_formula=bool(mineru.get("enable_formula", True)),
                    timeout_seconds=int(mineru.get("timeout_seconds", 600)),
                    poll_interval_seconds=int(mineru.get("poll_interval_seconds", 5)),
                    save_raw_result=bool(mineru.get("save_raw_result", True)),
                )
            pdf_parser_settings = PdfParserSettings(
                primary=_require_str(pdf_parser, "primary", "pdf_parser"),
                fallback_enabled=_require_bool(
                    pdf_parser, "fallback_enabled", "pdf_parser"
                ),
                fallback_provider=_require_str(
                    pdf_parser, "fallback_provider", "pdf_parser"
                ),
                fallback_threshold=_require_number(
                    pdf_parser, "fallback_threshold", "pdf_parser"
                ),
                mineru=mineru_settings,
            )

        vision_llm_settings = None
        if "vision_llm" in data:
            vision_llm = _require_mapping(data, "vision_llm", "settings")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vision_llm, "enabled", "vision_llm"),
                provider=_require_str(vision_llm, "provider", "vision_llm"),
                model=_require_str(vision_llm, "model", "vision_llm"),
                max_image_size=_require_int(vision_llm, "max_image_size", "vision_llm"),
                api_key=vision_llm.get("api_key"),
                api_version=vision_llm.get("api_version"),
                azure_endpoint=vision_llm.get("azure_endpoint"),
                deployment_name=vision_llm.get("deployment_name"),
                base_url=vision_llm.get("base_url"),
            )

        settings = cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                api_key=llm.get("api_key"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
                base_url=llm.get("base_url"),
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                api_key=embedding.get("api_key"),
                api_version=embedding.get("api_version"),
                azure_endpoint=embedding.get("azure_endpoint"),
                deployment_name=embedding.get("deployment_name"),
                base_url=embedding.get("base_url"),
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
                candidate_top_k=_optional_int(
                    rerank, "candidate_top_k", "rerank", 20
                ),
                timeout=_optional_number(rerank, "timeout", "rerank", 15.0),
                batch_size=_optional_int(rerank, "batch_size", "rerank", 16),
                max_length=_optional_int(rerank, "max_length", "rerank", 512),
                fallback_on_error=_optional_bool(
                    rerank, "fallback_on_error", "rerank", True
                ),
                execution_mode=_optional_str(
                    rerank, "execution_mode", "rerank", "in_process"
                ),
                startup_timeout=_optional_number(
                    rerank, "startup_timeout", "rerank", 60.0
                ),
                local_files_only=_optional_bool(
                    rerank, "local_files_only", "rerank", False
                ),
                cache_dir=_optional_str(
                    rerank,
                    "cache_dir",
                    "rerank",
                    "./data/models/huggingface/hub",
                ),
                api_key=rerank.get("api_key"),
                base_url=rerank.get("base_url"),
                max_retries=_optional_int(rerank, "max_retries", "rerank", 2),
                retry_base_delay=_optional_number(
                    rerank, "retry_base_delay", "rerank", 0.5
                ),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(observability, "structured_logging", "observability"),
            ),
            ingestion=ingestion_settings,
            vision_llm=vision_llm_settings,
            pdf_parser=pdf_parser_settings,
            citation_guard=citation_guard_settings,
        )

        return settings


def validate_settings(settings: Settings) -> None:
    """Validate settings and raise SettingsError if invalid."""

    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    if settings.rerank.top_k < 1:
        raise SettingsError("rerank.top_k must be greater than zero")
    if settings.rerank.candidate_top_k < settings.rerank.top_k:
        raise SettingsError("rerank.candidate_top_k must be >= rerank.top_k")
    if settings.rerank.timeout <= 0:
        raise SettingsError("rerank.timeout must be greater than zero")
    if settings.rerank.batch_size < 1:
        raise SettingsError("rerank.batch_size must be greater than zero")
    if settings.rerank.max_length < 1:
        raise SettingsError("rerank.max_length must be greater than zero")
    if settings.citation_guard is not None:
        guard = settings.citation_guard
        if guard.min_text_length < 1:
            raise SettingsError("citation_guard.min_text_length must be positive")
        if not 0 <= guard.weak_score_threshold <= 1:
            raise SettingsError(
                "citation_guard.weak_score_threshold must be between 0 and 1"
            )
        if not 0 <= guard.strong_score_threshold <= 1:
            raise SettingsError(
                "citation_guard.strong_score_threshold must be between 0 and 1"
            )
        if guard.strong_score_threshold < guard.weak_score_threshold:
            raise SettingsError(
                "citation_guard.strong_score_threshold must be >= weak_score_threshold"
            )
        if guard.min_strong_evidence < 1:
            raise SettingsError(
                "citation_guard.min_strong_evidence must be positive"
            )
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file and validate required fields.

    Args:
        path: Path to settings YAML.  Defaults to
            ``<repo>/config/settings.yaml`` (absolute, CWD-independent).
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    _load_dotenv_file(REPO_ROOT / ".env")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    data = _expand_env_placeholders(data or {})
    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings
