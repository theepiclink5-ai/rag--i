"""Query Traces page – browse query trace history with stage waterfall.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="query")
3. Detail view: stage waterfall + Dense vs Sparse comparison + Rerank delta
4. Per-trace Ragas evaluation button (LLM-as-Judge scoring)
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService


def _seconds_label(value_ms: Any) -> str:
    if not isinstance(value_ms, (int, float)):
        return "—"
    return f"{value_ms / 1000:.2f} s"

def render() -> None:
    """Render the Query Traces page."""
    st.header("查询追踪")

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        st.info("还没有查询记录，请先发起一次问答。")
        return

    # ── Keyword filter ─────────────────────────────────────────────
    keyword = st.text_input(
        "按问题关键词搜索",
        value="",
        key="qt_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(f"查询记录（{len(traces)}）")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # ── Expander title: show query text ────────────────────
        query_preview = (
            query_text[:40] + "…" if len(query_text) > 40 else query_text
        ) if query_text else "—"
        title_status = {
            "answered": "已回答",
            "insufficient_evidence": "证据不足",
            "error": "失败",
        }.get(meta.get("status"), "已记录")
        expander_title = (
            f"{title_status} | {query_preview} | {started[:19]}"
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            # Metrics are intentionally rendered before the technical trace.
            timings = svc.get_stage_timings(trace)
            latency = svc.get_latency_breakdown(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            dense_d = (stages_by_name.get("dense_retrieval", {}).get("data") or {})
            sparse_d = (stages_by_name.get("sparse_retrieval", {}).get("data") or {})
            fusion_d = (stages_by_name.get("fusion", {}).get("data") or {})
            rerank_d = (stages_by_name.get("rerank", {}).get("data") or {})

            dense_count = dense_d.get("result_count", 0)
            sparse_count = sparse_d.get("result_count", 0)
            fusion_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)

            _render_metric_summary(
                meta=meta,
                stages_by_name=stages_by_name,
                latency=latency,
                dense_count=dense_count,
                sparse_count=sparse_count,
                fusion_count=fusion_count,
                rerank_count=rerank_count,
            )

            # ── Diagnostic hints ───────────────────────────────
            _render_diagnostics(
                stages_by_name, dense_d, sparse_d, fusion_d, rerank_d,
                dense_count, sparse_count,
            )

            st.divider()
            answer_tab, timing_tab, retrieval_tab, technical_tab = st.tabs([
                "答案与引用", "阶段耗时", "检索详情", "技术信息",
            ])
            with answer_tab:
                _render_answer_summary(query_text, meta)
                if meta.get("mode") == "evaluation":
                    _render_recorded_evaluation(meta, stages_by_name, divider=False)
            with timing_tab:
                _render_stage_timings(timings, latency)
            with retrieval_tab:
                _render_stage_details(stages_by_name, idx)
            with technical_tab:
                st.markdown(f"**调用记录 ID：** `{trace_id}`")
                st.markdown(f"**来源：** `{source}`")
                st.markdown(f"**知识库：** `{meta.get('collection', '—')}`")
                st.markdown(f"**Top-K：** `{meta.get('top_k', '—')}`")
                st.markdown(f"**模型：** `{meta.get('model', '—')}`")


def _score_label(value: Any) -> str:
    return f"{float(value):.3f}" if isinstance(value, (int, float)) else "未评估"


def _render_metric_summary(
    *,
    meta: Dict[str, Any],
    stages_by_name: Dict[str, Any],
    latency: Dict[str, float],
    dense_count: int,
    sparse_count: int,
    fusion_count: int,
    rerank_count: int,
) -> None:
    """Render the decision-oriented metrics before all trace details."""
    evidence = stages_by_name.get("evidence_guard", {}).get("data") or {}
    metrics = meta.get("evaluation_metrics") or {}
    usage = meta.get("usage") or {}
    status_labels = {
        "answered": "已回答",
        "insufficient_evidence": "证据不足",
        "error": "执行失败",
    }
    evidence_labels = {
        "supported": "证据充分",
        "uncertain": "证据较弱",
        "no_evidence": "无有效证据",
    }
    status = status_labels.get(meta.get("status"), meta.get("status", "未知"))
    evidence_status = evidence_labels.get(
        evidence.get("status"), evidence.get("status", "未记录")
    )
    citations = meta.get("citations") or []
    top_score = evidence.get("top_rerank_score", meta.get("top_rerank_score"))
    evaluation_label = (
        _seconds_label(latency["evaluation_ms"])
        if "ragas_evaluation" in stages_by_name else "未评估"
    )

    st.markdown("#### 本次问答概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("回答状态", status)
    c2.metric("证据状态", evidence_status)
    citable_count = evidence.get("citable_count", len(meta.get("final_results") or []))
    rejected_count = evidence.get("rejected_count", meta.get("evidence_rejected_count", 0))
    c3.metric("有效 / 淘汰证据", f"{citable_count} / {rejected_count}")
    c4.metric("引用数量", len(citations))

    st.markdown("##### 质量指标")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("忠实度", _score_label(metrics.get("faithfulness")), help="答案中的陈述是否得到参考文本支持。")
    q2.metric("答案相关性", _score_label(metrics.get("answer_relevancy")), help="答案是否切中用户问题。")
    q3.metric("上下文精确度", _score_label(metrics.get("context_precision")), help="排在前面的参考片段是否真正有用。")
    q4.metric("最高相关分", _score_label(top_score), help="Cross-Encoder 重排后的最高相关性分数。")

    st.markdown("##### 检索与资源")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("语义 / 关键词召回", f"{dense_count} / {sparse_count}")
    r2.metric("融合 / 重排保留", f"{fusion_count or dense_count + sparse_count} / {rerank_count or '—'}")
    r3.metric("总 Token", usage.get("total_tokens", "—"))
    prompt_tokens = usage.get("prompt_tokens", "—")
    completion_tokens = usage.get("completion_tokens", "—")
    r4.metric("输入 / 输出 Token", f"{prompt_tokens} / {completion_tokens}")

    st.markdown("##### 性能指标")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("回答耗时", _seconds_label(latency["answer_ready_ms"]))
    p2.metric("评估耗时", evaluation_label)
    p3.metric("完整耗时", _seconds_label(latency["total_ms"]))
    p4.metric("启动及其他", _seconds_label(latency["other_answer_ms"]))


def _render_answer_summary(query: str, meta: Dict[str, Any]) -> None:
    st.markdown("#### 问题")
    st.markdown(f"> {query or '—'}")
    st.markdown("#### 最终答案")
    st.markdown(meta.get("answer") or "该记录没有保存最终答案。")
    citations = meta.get("citations") or []
    if citations:
        st.markdown("#### 引用")
        for citation in citations:
            index = citation.get("index", "—")
            source = citation.get("source", "未知来源")
            score = citation.get("score")
            score_text = f"，相关分 {score:.4f}" if isinstance(score, (int, float)) else ""
            st.caption(f"[{index}] {source}{score_text}")


def _render_stage_timings(
    timings: List[Dict[str, Any]], latency: Dict[str, float]
) -> None:
    stage_labels = {
        "reranker_startup": "重排模型启动",
        "initialization": "查询组件初始化",
        "query_processing": "问题处理",
        "dense_retrieval": "语义检索",
        "sparse_retrieval": "关键词检索",
        "fusion": "RRF 融合",
        "rerank": "Cross-Encoder 重排",
        "evidence_guard": "引用约束检查",
        "llm_generation": "生成答案",
        "ragas_evaluation": "Ragas 评估",
    }
    main_timings = [t for t in timings if t["stage_name"] in stage_labels]
    if not main_timings:
        st.info("该记录没有阶段耗时数据。")
        return
    chart_data = {
        stage_labels[t["stage_name"]]: t["elapsed_ms"] / 1000
        for t in main_timings
    }
    st.bar_chart(chart_data, horizontal=True)
    st.table([
        {
            "阶段": stage_labels[t["stage_name"]],
            "耗时（秒）": round(t["elapsed_ms"] / 1000, 3),
        }
        for t in main_timings
    ])
    st.caption(
        "语义检索和关键词检索并行执行，对总耗时的贡献约为 "
        f"{latency['retrieval_wall_ms'] / 1000:.3f} 秒，不能将两行直接相加。"
    )


def _render_stage_details(stages_by_name: Dict[str, Any], trace_idx: int) -> None:
    tab_defs = []
    if "query_processing" in stages_by_name:
        tab_defs.append(("问题处理", "query_processing"))
    if "dense_retrieval" in stages_by_name:
        tab_defs.append(("语义检索", "dense_retrieval"))
    if "sparse_retrieval" in stages_by_name:
        tab_defs.append(("关键词检索", "sparse_retrieval"))
    if "fusion" in stages_by_name:
        tab_defs.append(("RRF 融合", "fusion"))
    if "rerank" in stages_by_name:
        tab_defs.append(("重排", "rerank"))
    if not tab_defs:
        st.info("该记录没有检索阶段详情。")
        return
    tabs = st.tabs([label for label, _ in tab_defs])
    for tab, (_, key) in zip(tabs, tab_defs):
        with tab:
            stage = stages_by_name[key]
            data = stage.get("data", {})
            elapsed = stage.get("elapsed_ms")
            if elapsed is not None:
                st.caption(f"耗时：{elapsed / 1000:.3f} 秒")
            if key == "query_processing":
                _render_query_processing_stage(data)
            elif key == "dense_retrieval":
                _render_retrieval_stage(data, "语义检索", trace_idx=trace_idx)
            elif key == "sparse_retrieval":
                _render_retrieval_stage(data, "关键词检索", trace_idx=trace_idx)
            elif key == "fusion":
                _render_fusion_stage(data, trace_idx=trace_idx)
            elif key == "rerank":
                _render_rerank_stage(data, trace_idx=trace_idx)


def _render_recorded_evaluation(
    meta: Dict[str, Any],
    stages_by_name: Dict[str, Any],
    *,
    divider: bool = True,
) -> None:
    """Render metrics produced by the evaluated request itself."""
    if divider:
        st.divider()
    st.markdown("#### 详细质量评估")
    status = meta.get("evaluation_status", "unknown")
    metrics = meta.get("evaluation_metrics") or {}

    if status == "completed" and metrics:
        labels = {
            "faithfulness": "忠实度",
            "answer_relevancy": "答案相关性",
            "context_precision": "上下文精确度",
        }
        columns = st.columns(len(metrics))
        for column, (name, value) in zip(columns, sorted(metrics.items())):
            with column:
                st.metric(labels.get(name, name), f"{float(value):.3f}")

        evaluation_stage = stages_by_name.get("ragas_evaluation", {})
        elapsed = evaluation_stage.get("elapsed_ms")
        if elapsed is not None:
            st.caption(f"Ragas 评估耗时：{elapsed / 1000:.2f} 秒")
    elif status == "error":
        st.error(f"评估失败：{meta.get('evaluation_error', '未知错误')}")
    elif status == "skipped":
        st.info(
            "评估已跳过，回答流程状态："
            f"{meta.get('evaluation_reason', 'unknown')}"
        )
    else:
        st.info("这次请求没有详细评估结果。")


def _render_diagnostics(
    stages_by_name: Dict[str, Any],
    dense_d: Dict[str, Any],
    sparse_d: Dict[str, Any],
    fusion_d: Dict[str, Any],
    rerank_d: Dict[str, Any],
    dense_count: int,
    sparse_count: int,
) -> None:
    """Render diagnostic hints about missing or errored pipeline stages."""
    hints: list = []

    # Dense errors
    dense_err = dense_d.get("error", "")
    if dense_err:
        hints.append(("error", f"**语义检索失败：** {dense_err}"))
    elif dense_count == 0 and "dense_retrieval" in stages_by_name:
        hints.append(("warning", "语义检索返回 **0 个结果**，请检查知识库是否已经建立索引。"))

    # Sparse errors / empty
    sparse_err = sparse_d.get("error", "")
    if sparse_err:
        hints.append(("error", f"**关键词检索失败：** {sparse_err}"))
    elif sparse_count == 0 and "sparse_retrieval" in stages_by_name:
        hints.append((
            "warning",
            "关键词检索（BM25）返回 **0 个结果**，该知识库的 BM25 索引可能为空或尚未建立。",
        ))

    # Fusion missing
    if "fusion" not in stages_by_name:
        if dense_count > 0 and sparse_count > 0:
            hints.append(("info", "两路检索均有结果，但没有记录 RRF 融合阶段。"))
        elif dense_count == 0 or sparse_count == 0:
            only_source = "语义检索" if dense_count > 0 else ("关键词检索" if sparse_count > 0 else "两路检索")
            hints.append((
                "info",
                f"**已跳过 RRF 融合：** 只有{only_source}返回结果；两路都有结果时才能进行融合。",
            ))

    # Rerank missing
    if "rerank" not in stages_by_name:
        if dense_count > 0 or sparse_count > 0:
            hints.append((
                "info",
                "**已跳过重排：** 重排模型尚未启用或配置。",
            ))

    # All results empty
    if dense_count == 0 and sparse_count == 0:
        hints.append((
            "warning",
            "**没有检索到结果。** 知识库可能为空，或者问题与已有内容不匹配。",
        ))

    # Render hints
    for level, msg in hints:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def _extract_pipeline_chunks(
    timings: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract chunk lists from each pipeline stage."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for stage in timings:
        name = stage.get("stage_name", "")
        data = stage.get("data") or {}
        chunks = data.get("chunks")
        if chunks and isinstance(chunks, list):
            result[name] = chunks
    final = meta.get("final_results") or meta.get("results")
    if final and isinstance(final, list):
        result["final"] = final
    return result


# ═══════════════════════════════════════════════════════════════
# Per-stage renderers
# ═══════════════════════════════════════════════════════════════

def _render_query_processing_stage(data: Dict[str, Any]) -> None:
    """Render Query Processing stage: original query → keywords."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**原始问题**")
        st.info(data.get("original_query", "—"))
    with c2:
        st.markdown("**处理方法**")
        st.code(data.get("method", "—"))

    keywords = data.get("keywords", [])
    if keywords:
        st.markdown("**提取的关键词**")
        st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning("没有提取到关键词。")


def _render_retrieval_stage(data: Dict[str, Any], label: str, *, trace_idx: int = 0) -> None:
    """Render Dense or Sparse retrieval stage: method, counts, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("检索方法", data.get("method", "—"))
    with c2:
        extra = data.get("provider", data.get("keyword_count", "—"))
        extra_label = "服务提供方" if "provider" in data else "关键词数量"
        st.metric(extra_label, extra)
    with c3:
        st.metric("结果数量", data.get("result_count", 0))

    st.markdown(f"**请求数量上限：** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"{label.lower().replace(' ', '_')}_chunk_{trace_idx}")
    else:
        st.info(f"{label}没有返回结果。")


def _render_fusion_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Fusion (RRF) stage: input lists, fused result count, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("融合方法", data.get("method", "rrf"))
    with c2:
        st.metric("输入列表", data.get("input_lists", "—"))
    with c3:
        st.metric("融合结果", data.get("result_count", 0))

    st.markdown(f"**Top-K:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"fusion_chunk_{trace_idx}")
    else:
        st.info("没有融合结果。")


def _render_rerank_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """Render Rerank stage: method, input/output counts, reranked chunk list."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("重排方法", data.get("method", "—"))
    with c2:
        st.metric("服务提供方", data.get("provider", "—"))
    with c3:
        st.metric("输入数量", data.get("input_count", "—"))
    with c4:
        st.metric("输出数量", data.get("output_count", "—"))

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"rerank_chunk_{trace_idx}")
    else:
        st.info("没有重排结果。")


def _render_chunk_list(chunks: List[Dict[str, Any]], prefix: str = "chunk") -> None:
    """Render a list of chunk dicts as a compact, readable table with expandable text."""
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        # Colour-coded score indicator
        if score >= 0.8:
            score_bar = "🟢"
        elif score >= 0.5:
            score_bar = "🟡"
        else:
            score_bar = "🔴"

        header = f"{score_bar} **#{ci + 1}** — 分数：`{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"片段 ID：`{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"来源：`{source}`")
            # Show chunk text (scrollable)
            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption("_没有可显示的文本_ ")


def _find_stage(timings, name):
    """Find a stage dict by name, or None."""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
