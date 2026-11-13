"""评估面板 page – run evaluations and view metrics.

Layout:
1. 配置 section: select evaluator backend, golden test set, top_k
2. Run button with progress indicator
3. Results section: aggregate metrics, per-query detail table
4. Optional: historical evaluation results comparison
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

try:
    from src.observability.dashboard.services.data_service import DataService
except ImportError:
    DataService = None  # type: ignore[assignment,misc]

# Default golden test set location
DEFAULT_GOLDEN_SET = Path("tests/fixtures/golden_test_set.json")
# Evaluation results history file
EVAL_HISTORY_PATH = Path("logs/eval_history.jsonl")
RAGAS_SAMPLE_PATH = Path("logs/ragas_sample_10.json")

METRIC_LABELS = {
    "hit_rate": "命中率@5",
    "mrr": "平均倒数排名",
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "answer_correctness": "答案正确性",
    "answer_correctness_verdict": "答案正确性判定",
    "refusal_accuracy": "拒答准确率",
    "p95_latency": "P95 总延迟",
}

METRIC_ENGLISH = {
    "hit_rate": "Hit Rate@5",
    "mrr": "MRR",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
    "answer_correctness_verdict": "Correctness Verdict",
    "refusal_accuracy": "Refusal Accuracy",
    "p95_latency": "P95 Total Latency",
}

def render() -> None:
    """Render the 评估面板 page."""
    st.header("📏 评估面板")
    st.markdown(
        "\u57fa\u4e8e **golden test set** \u8fd0\u884c\u8bc4\u4f30\uff0c\u7528\u4e8e\u5ea6\u91cf\u68c0\u7d22 "
        "\u4e0e\u751f\u6210\u8d28\u91cf\u3002\u7ed3\u679c\u5305\u542b\u5355\u6761 Query \u660e\u7ec6\u4ee5\u53ca "
        "\u805a\u5408\u6307\u6807\u3002"
    )

    _render_quality_overview()

    # ── 配置 Section ──────────────────────────────────────
    st.subheader("⚙️ 配置")

    col1, col2, col3 = st.columns(3)

    with col1:
        backend = st.selectbox(
            "\u8bc4\u4f30\u540e\u7aef",
            options=["custom", "ragas", "composite"],
            index=0,
            key="eval_backend",
            help="选择要使用的 evaluator backend。",
        )

    # Show info/warning based on selected backend
    if backend in ("custom", "composite"):
        st.info(
            "ℹ️ **Custom Evaluator** 尚未完成数据集准备，当前仅为预留接口。"
            "Custom Evaluator 需要在 Golden Test Set 中填写 `expected_chunk_ids` 或 `expected_sources` "
            "作为 ground truth 才能计算 hit_rate / MRR 指标。"
            "目前建议使用 **ragas** 后端进行评估。",
            icon="🚧",
        )

    with col2:
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=50,
            value=10,
            key="eval_top_k",
            help="每个 query 要检索的 Chunk 数量。",
        )

    with col3:
        _available_collections = []
        if DataService is not None:
            try:
                _available_collections = DataService().list_collections()
            except Exception:
                _available_collections = ["default"]
        if not _available_collections:
            _available_collections = ["default"]
        collection = st.selectbox(
            "Collection",
            options=_available_collections,
            index=0,
            key="eval_collection",
            help="选择要评估的 Collection 分组。",
        )

    # ── 上传 JSON 问答对数据集 ──────────────────────────────
    uploaded_file = st.file_uploader(
        "上传问答对数据集（JSON）",
        type=["json"],
        key="eval_json_upload",
        help="上传包含 test_cases 数组的 JSON 文件，每个元素需包含 query 字段。",
    )
    _schema_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "docs" / "evaluation_dataset_schema.md"
    _schema_col1, _schema_col2 = st.columns([3, 1])
    with _schema_col1:
        st.caption(
            "📖 必填字段：`query`；可选：`reference_answer`、`expected_chunk_ids`、`expected_sources`。"
        )
    with _schema_col2:
        if _schema_path.exists():
            st.download_button(
                label="📥 下载格式规范",
                data=_schema_path.read_text(encoding="utf-8"),
                file_name="evaluation_dataset_schema.md",
                mime="text/markdown",
                key="eval_schema_download",
            )
        else:
            st.caption("⚠️ 格式规范文件未找到")
    golden_path = DEFAULT_GOLDEN_SET  # fallback
    _upload_valid = False
    if uploaded_file is not None:
        try:
            _uploaded_data = json.loads(uploaded_file.getvalue().decode("utf-8"))
            # Normalize to test_cases list
            _raw_cases = []
            if isinstance(_uploaded_data, dict) and "test_cases" in _uploaded_data:
                _raw_cases = _uploaded_data["test_cases"]
            elif isinstance(_uploaded_data, list):
                _raw_cases = _uploaded_data
            else:
                st.error("❌ JSON 格式不正确：需要包含 test_cases 数组或顶层为数组。")

            # Validate each test case before accepting
            if _raw_cases:
                _validation_errors = []
                for _idx, _tc in enumerate(_raw_cases):
                    if not isinstance(_tc, dict):
                        _validation_errors.append(f"第 {_idx + 1} 条：不是有效的 JSON 对象")
                        continue
                    if "query" not in _tc:
                        _validation_errors.append(f"第 {_idx + 1} 条：缺少必填字段 `query`")
                    elif not isinstance(_tc["query"], str) or not _tc["query"].strip():
                        _validation_errors.append(f"第 {_idx + 1} 条：`query` 必须为非空字符串")
                    if "expected_chunk_ids" in _tc and not isinstance(_tc["expected_chunk_ids"], list):
                        _validation_errors.append(f"第 {_idx + 1} 条：`expected_chunk_ids` 必须为数组")
                    if "expected_sources" in _tc and not isinstance(_tc["expected_sources"], list):
                        _validation_errors.append(f"第 {_idx + 1} 条：`expected_sources` 必须为数组")

                if _validation_errors:
                    _upload_valid = False
                    st.error(f"❌ 数据集校验失败（共 {len(_validation_errors)} 个问题）：")
                    for _err in _validation_errors[:20]:
                        st.text(f"  • {_err}")
                    if len(_validation_errors) > 20:
                        st.caption(f"…还有 {len(_validation_errors) - 20} 个问题未显示")
                else:
                    _upload_valid = True
                    _normalized = {"test_cases": _raw_cases}
                    _tmp_path = Path("/tmp/eval_uploaded_test_set.json")
                    _tmp_path.write_text(json.dumps(_normalized, ensure_ascii=False), encoding="utf-8")
                    golden_path = _tmp_path
                    _ref_count = sum(1 for _tc in _raw_cases if _tc.get("reference_answer"))
                    st.success(
                        f"✅ 校验通过，已加载 {len(_raw_cases)} 条测试用例"
                        f"（其中 {_ref_count} 条含参考答案）。"
                    )
            else:
                st.error("❌ JSON 格式不正确：需要包含 test_cases 数组或顶层为数组。")
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON 解析失败：{e}")
    else:
        if golden_path.exists():
            st.info(f"ℹ️ 未上传文件，将使用默认测试集：`{golden_path}`")
        else:
            st.warning(
                "⚠️ **请上传 JSON 问答对数据集。** "
                'JSON 格式示例：`{"test_cases": [{"query": "...", "reference_answer": "..."}]}`'
            )
    # ── 数据集预览 ──────────────────────────────────────────
    user_answers: Dict[int, str] = {}
    if _upload_valid or golden_path.exists():
        try:
            _test_cases = _load_golden_queries(golden_path)
            if _test_cases:
                st.divider()
                st.subheader("📋 数据集预览")
                st.caption(f"共 {len(_test_cases)} 条问答对。包含 reference_answer 的用例将自动用于评估。")
                for tc_idx, tc in enumerate(_test_cases):
                    ref_ans = tc.get("reference_answer", "")
                    if ref_ans and str(ref_ans).strip():
                        user_answers[tc_idx] = str(ref_ans).strip()
                if user_answers:
                    st.success(f"✅ 数据集中已有 {len(user_answers)}/{len(_test_cases)} 条包含参考答案。")
                else:
                    st.info("ℹ️ 数据集中无 reference_answer 字段，评估时将使用检索片段拼接作为回答。")
        except Exception as exc:
            st.warning(f"无法预览数据集: {exc}")

    # ── 开始评估 ─────────────────────────────────────────────
    st.divider()

    _can_run = _upload_valid or golden_path.exists()
    run_clicked = st.button(
        "▶️  开始评估",
        type="primary",
        key="eval_run_btn",
        disabled=not _can_run,
    )

    if run_clicked:
        _run_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=int(top_k),
            collection=collection.strip() or None,
            user_answers=user_answers if user_answers else None,
        )

    # ── Historical Results ─────────────────────────────────────────
    st.divider()
    _render_history()


def _render_quality_overview() -> None:
    """Show the latest available quality signals without inventing missing scores."""
    metrics, sources = _load_quality_snapshot()
    st.divider()
    st.subheader("质量指标总览")
    st.caption("汇总当前已有的离线评测和线上调用记录；待评估表示尚无对应数据。")

    groups = [
        ("检索质量", ["hit_rate", "mrr"]),
        ("上下文质量", ["context_precision", "context_recall"]),
        (
            "回答质量",
            ["faithfulness", "answer_relevancy", "answer_correctness"],
        ),
        ("安全与性能", ["refusal_accuracy", "p95_latency"]),
    ]
    for group_name, names in groups:
        st.markdown(f"**{group_name}**")
        columns = st.columns(3)
        for index, name in enumerate(names):
            value = metrics.get(name)
            display = _format_quality_value(name, value)
            with columns[index]:
                st.metric(METRIC_LABELS[name], display)
                st.caption(
                    f"{METRIC_ENGLISH[name]} · {sources.get(name, '暂无数据')}"
                )


def _load_quality_snapshot() -> tuple[Dict[str, float], Dict[str, str]]:
    metrics: Dict[str, float] = {}
    sources: Dict[str, str] = {}

    if RAGAS_SAMPLE_PATH.exists():
        try:
            with RAGAS_SAMPLE_PATH.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            aggregate = report.get("aggregate", {})
            sample_count = int(aggregate.get("total", report.get("limit", 0)))
            source_label = f"Ragas 样本 {sample_count} 条"
            for name, value in aggregate.get("average_metrics", {}).items():
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)
                    sources[name] = source_label
        except (OSError, ValueError, TypeError):
            logger.exception("Failed to load Ragas sample report")

    history = _load_history()
    # Metrics to load from evaluation history (in priority order)
    _history_metrics = (
        "hit_rate", "mrr",
        "context_recall", "answer_correctness", "answer_correctness_verdict",
        "faithfulness", "answer_relevancy", "context_precision",
    )
    for entry in reversed(history):
        aggregate_metrics = entry.get("aggregate_metrics", {})
        for name in _history_metrics:
            value = aggregate_metrics.get(name)
            if name not in metrics and isinstance(value, (int, float)):
                metrics[name] = float(value)
                count = int(entry.get("query_count", 0))
                sources[name] = f"Golden Set {count} 条"

    # Compute p95 from the latest evaluation history's query_results
    if history:
        latest = history[-1]
        _qr = latest.get("query_results", [])
        _latencies = sorted(
            float(qr.get("elapsed_ms", 0))
            for qr in _qr
            if isinstance(qr.get("elapsed_ms"), (int, float)) and qr.get("elapsed_ms", 0) > 0
        )
        if _latencies:
            _p95_idx = max(0, math.ceil(len(_latencies) * 0.95) - 1)
            metrics["p95_latency"] = _latencies[_p95_idx]
            sources["p95_latency"] = f"最近评估 {latest.get('query_count', 0)} 条"

    return metrics, sources


def _format_quality_value(name: str, value: Optional[float]) -> str:
    if value is None:
        return "待评估"
    if name == "p95_latency":
        return f"{value / 1000:.2f} s"
    return f"{value:.3f}"


def _run_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
) -> None:
    """Execute an evaluation run and display results with live progress."""
    # Pre-load test cases count for progress bar
    try:
        with golden_path.open("r", encoding="utf-8") as _f:
            _tc_data = json.load(_f)
        _total_cases = len(_tc_data.get("test_cases", []))
    except Exception:
        _total_cases = 0

    progress_bar = st.progress(0, text="正在初始化评估…")
    status_text = st.empty()

    def _on_progress(current: int, total: int, query: str, elapsed_ms: float) -> None:
        pct = current / total if total > 0 else 0
        progress_bar.progress(pct, text=f"评估中 {current}/{total} ({pct:.0%})")
        status_text.caption(
            f"⏱️ 当前: {elapsed_ms/1000:.1f}s | "
            f"已完成: {current}/{total} | "
            f"问题: {query[:60]}{'…' if len(query) > 60 else ''}"
        )

    try:
        report_dict = _execute_evaluation(
            backend=backend,
            golden_path=golden_path,
            top_k=top_k,
            collection=collection,
            user_answers=user_answers,
            progress_callback=_on_progress,
        )
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ 评估失败：{exc}")
        logger.exception("Evaluation failed")
        return

    progress_bar.progress(1.0, text="✅ 评估完成")
    status_text.empty()
    # ── Display results ────────────────────────────────────────────
    st.success("✅ 评估完成！")

    _render_aggregate_metrics(report_dict)
    _render_query_details(report_dict)

    # Save to history
    _save_to_history(report_dict)


def _execute_evaluation(
    backend: str,
    golden_path: Path,
    top_k: int,
    collection: Optional[str],
    user_answers: Optional[Dict[int, str]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run the evaluation pipeline and return the report dict.

    This function imports heavy dependencies lazily to keep the
    dashboard responsive when the page is not used.
    """
    from dataclasses import replace as dc_replace

    from src.core.settings import load_settings
    from src.libs.evaluator.evaluator_factory import EvaluatorFactory
    from src.observability.evaluation.eval_runner import EvalRunner, load_test_set

    settings = load_settings()

    # Override evaluator provider from UI selection — build a new full
    # Settings object so that RagasEvaluator can still access .llm / .embedding.
    eval_settings = settings.evaluation
    # When using ragas backend, override metrics to Ragas-supported set
    # so that filtering in RagasEvaluator doesn't leave an empty/useless list.
    _ragas_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    if backend == "ragas":
        _effective_metrics = _ragas_metrics
    else:
        _effective_metrics = eval_settings.metrics if hasattr(eval_settings, "metrics") else []
    overridden_eval = type(eval_settings)(
        enabled=True,
        provider=backend,
        metrics=_effective_metrics,
    )
    # Replace only the evaluation sub-config in the full settings
    settings_with_override = dc_replace(settings, evaluation=overridden_eval)

    evaluator = EvaluatorFactory.create(settings_with_override)

    # Try to create HybridSearch (optional – works without if not configured)
    target_collection = collection or "default"
    hybrid_search = _try_create_hybrid_search(settings, target_collection)

    # Create reranker if enabled
    reranker = None
    try:
        from src.core.query_engine.reranker import create_core_reranker
        reranker = create_core_reranker(settings=settings)
        if not reranker.is_enabled:
            reranker = None
    except Exception as exc:
        logger.warning("Could not create reranker: %s", exc)

    # Build answer_override map: index → user-provided answer text
    # EvalRunner will use these instead of auto-generating from chunks.
    # NOTE: user_answers (reference_answer from golden set) are used as
    # ground_truth for context_recall / answer_correctness, NOT as answer overrides.
    # LLM should always generate the answer for Ragas evaluation.
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
        answer_overrides=None,  # Never override; let LLM generate answers
        reranker=reranker,
    )

    report = runner.run(
        test_set_path=golden_path,
        top_k=top_k,
        collection=collection,
        progress_callback=progress_callback,
    )

    return report.to_dict()


def _try_create_hybrid_search(settings: Any, collection: str = "default") -> Any:
    """Attempt to create a HybridSearch instance.

    Returns None if required dependencies are not available
    (e.g., no indexed data).
    """
    try:
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        return create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
    except Exception as exc:
        logger.warning("Could not create HybridSearch: %s", exc)
        return None


def _render_aggregate_metrics(report: Dict[str, Any]) -> None:
    """Display aggregate metrics as metric cards, including p95 from this run."""
    st.subheader("📊 聚合指标")
    agg = dict(report.get("aggregate_metrics", {}))

    # Compute p95 latency from this evaluation's query_results
    query_results = report.get("query_results", [])
    latencies = sorted(
        float(qr.get("elapsed_ms", 0))
        for qr in query_results
        if isinstance(qr.get("elapsed_ms"), (int, float)) and qr.get("elapsed_ms", 0) > 0
    )
    if latencies:
        p95_idx = max(0, math.ceil(len(latencies) * 0.95) - 1)
        agg["p95_latency"] = latencies[p95_idx]

    if not agg:
        st.info("暂无聚合指标结果。")
        return
    cols = st.columns(min(len(agg), 4))
    for idx, (name, value) in enumerate(sorted(agg.items())):
        display = f"{value / 1000:.2f}s" if name == "p95_latency" else f"{value:.4f}"
        with cols[idx % len(cols)]:
            st.metric(
                label=METRIC_LABELS.get(name, name.replace("_", " ").title()),
                value=display,
            )
    st.caption(
        f"Evaluator： **{report.get('evaluator_name', '-')}** · "
        f"Queries： **{report.get('query_count', 0)}** · "
        f"总耗时： **{report.get('total_elapsed_ms', 0):.0f} ms**"
    )
def _render_query_details(report: Dict[str, Any]) -> None:
    """Display per-query evaluation results in an expandable table."""
    st.subheader("🔍 单条 Query 明细")

    query_results = report.get("query_results", [])
    if not query_results:
        st.info("\u6682\u65e0\u5355\u6761 Query \u8bc4\u4f30\u7ed3\u679c\u3002")
        return

    for idx, qr in enumerate(query_results):
        query = qr.get("query", "—")
        elapsed = qr.get("elapsed_ms", 0)
        metrics = qr.get("metrics", {})

        # Build metric summary for the expander label
        metric_summary = " · ".join(
            f"{k}: {v:.3f}" for k, v in sorted(metrics.items())
        )
        if not metric_summary:
            metric_summary = "\u65e0\u6307\u6807"

        with st.expander(
            f"**Q{idx + 1}**: {query[:80]} — {elapsed:.0f} ms — {metric_summary}",
            expanded=False,
        ):
            # Metrics
            if metrics:
                mcols = st.columns(min(len(metrics), 4))
                for midx, (mname, mval) in enumerate(sorted(metrics.items())):
                    with mcols[midx % len(mcols)]:
                        st.metric(mname, f"{mval:.4f}")

            # Retrieved chunks
            chunks = qr.get("retrieved_chunk_ids", [])
            if chunks:
                st.markdown(f"**Retrieved Chunks\uff1a{len(chunks)} \u6761**")
                st.code(", ".join(chunks[:20]), language=None)

            # Generated answer
            answer = qr.get("generated_answer")
            if answer:
                st.markdown("**\u751f\u6210\u56de\u7b54**")
                st.text(answer[:500])


def _render_history() -> None:
    """Display historical evaluation results for comparison."""
    st.subheader("📈 评估历史")

    history = _load_history()
    if not history:
        st.info(
            "**\u6682\u65e0\u8bc4\u4f30\u5386\u53f2\u8bb0\u5f55\u3002** "
            "请先在上方配置 evaluator，然后点击开始评估按钮。"
            "\u5b8c\u6210\u4e4b\u540e\uff0c\u8fd9\u91cc\u4f1a\u663e\u793a\u6700\u8fd1\u7684\u8bc4\u4f30\u8fd0\u884c\u7ed3\u679c\u3002"
        )
        return

    # Show recent runs as a table
    rows = []
    for entry in history[-10:]:  # last 10 runs
        rows.append(
            {
                "\u65f6\u95f4": entry.get("timestamp", "\u2014"),
                "Evaluator": entry.get("evaluator_name", "\u2014"),
                "Queries": entry.get("query_count", 0),
                "\u8017\u65f6 (ms)": round(entry.get("total_elapsed_ms", 0)),
                **{
                    k: round(v, 4)
                    for k, v in entry.get("aggregate_metrics", {}).items()
                },
            }
        )

    st.dataframe(rows, use_container_width=True)


def _save_to_history(report: Dict[str, Any]) -> None:
    """Append an evaluation report to the history file."""
    try:
        EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with EVAL_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save evaluation history: %s", exc)


def _load_history() -> List[Dict[str, Any]]:
    """Load evaluation history from JSONL file."""
    if not EVAL_HISTORY_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with EVAL_HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        logger.warning("Failed to load evaluation history: %s", exc)

    return entries


def _load_golden_queries(golden_path: Path) -> List[Dict[str, Any]]:
    """Load test cases from golden test set for display in the UI.

    Returns list of dicts with at least 'query' and optionally
    'reference_answer' keys.
    """
    with golden_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])
