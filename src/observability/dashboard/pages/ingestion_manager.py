"""Ingestion Manager page - upload files, trigger ingestion, delete documents."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


_STAGE_LABELS = {
    "integrity": "正在执行文件完整性校验...",
    "load": "正在加载文档...",
    "split": "正在切分文档为 Chunks...",
    "transform": "正在处理 Chunks（LLM refine + enrich）...",
    "embed": "正在生成 Embedding...",
    "upsert": "正在写入向量存储...",
}


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """Save the uploaded file to a temp location and run the pipeline."""
    from src.core.settings import load_settings
    from src.core.trace import TraceCollector, TraceContext
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()

    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection)
        pipeline.run(file_path=tmp_path, trace=trace, on_progress=on_progress)
        progress_bar.progress(1.0, text="导入完成")
        status_text.success(
            f"已将 **{uploaded_file.name}** 导入到 Collection **{collection}**。"
        )
    except Exception as exc:
        status_text.error(f"导入失败：{exc}")
    finally:
        TraceCollector().collect(trace)
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """Render the Ingestion Manager page."""
    st.header("📥 导入管理")
    st.subheader("上传与导入")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "选择要导入的文件",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key="ingest_uploader",
        )
    with col2:
        collection = st.text_input("Collection", value="default", key="ingest_collection")

    if uploaded_files and st.button("开始导入", key="btn_ingest"):
        target_collection = collection.strip() or "default"
        total_files = len(uploaded_files)
        overall_progress = st.progress(0, text=f"准备导入 {total_files} 个文件...")

        for idx, uploaded in enumerate(uploaded_files, start=1):
            st.markdown(f"**[{idx}/{total_files}] {uploaded.name}**")
            progress_bar = st.progress(0, text="准备中...")
            status_text = st.empty()
            _run_ingestion(uploaded, target_collection, progress_bar, status_text)
            overall_progress.progress(
                idx / total_files,
                text=f"已处理 {idx}/{total_files} 个文件",
            )

        st.success(f"批量导入完成：已处理 {total_files} 个文件。")

    st.divider()
    st.subheader("已导入文档")

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"加载已导入文档失败：{exc}")
        return

    if not docs:
        st.info("暂无已导入文档。请上传 PDF、TXT、MD 或 DOCX 文件开始导入。")
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** - Collection: `{doc.get('collection', '-')}` | "
                f"Chunks: {doc['chunk_count']} | 图片: {doc['image_count']}"
            )
        with col_btn:
            if st.button("删除文档", key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            f"删除完成：已删除 {result.chunks_deleted} 个 Chunks，"
                            f" {result.images_deleted} 张图片。"
                        )
                        st.rerun()
                    else:
                        st.warning(f"删除未完全成功，发生错误：{result.errors}")
                except Exception as exc:
                    st.error(f"删除失败：{exc}")
