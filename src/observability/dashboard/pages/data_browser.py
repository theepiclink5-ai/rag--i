"""Data Browser page - browse ingested documents, chunks, and images."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


HEADER = "🔍 数据浏览"
DANGER_ZONE = "⚠️ 危险操作区"
CLEAR_ALL = "🗑️ 清空全部数据"
CONFIRM_CLEAR = "✅ 确认全部删除"
CANCEL_CLEAR = "❌ 取消"
DOC_LIST = "📄 文档列表"
METADATA = "📋 Metadata"
IMAGES = "🖼️ 图片"
CHUNKS = "📦 Chunks"


def render() -> None:
    """Render the Data Browser page."""
    st.header(HEADER)

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"初始化 DataService 失败：{exc}")
        return

    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")

    collection = st.selectbox(
        "Collection",
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    st.divider()
    with st.expander(DANGER_ZONE, expanded=False):
        st.warning(
            "这会**永久删除**所有数据："
            "ChromaDB collections、BM25 indexes、图片、导入历史和 Trace logs。"
        )
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            if st.button(CLEAR_ALL, type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error("确认继续吗？这个操作无法撤销。")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button(CONFIRM_CLEAR, key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            f"清理完成，但出现 {len(result['errors'])} 个错误："
                            + "; ".join(result["errors"])
                        )
                    else:
                        st.success(
                            f"所有数据已清空，"
                            f"共删除 {result['collections_deleted']} 个 Collection。"
                        )
                    st.rerun()
            with c2:
                if st.button(CANCEL_CLEAR, key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"加载文档失败：{exc}")
        return

    if not docs:
        st.info(
            "当前 Collection 中没有文档。"
            "请前往导入管理页面上传并导入文件，"
            "或在上方切换到其他 Collection。"
        )
        return

    st.subheader(f"{DOC_LIST} ({len(docs)})")

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = (
            f"📄 {source_name} - {doc['chunk_count']} 个 Chunks | "
            f"{doc['image_count']} 张图片"
        )
        with st.expander(label, expanded=(len(docs) == 1)):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chunks", doc["chunk_count"])
            col_b.metric("图片", doc["image_count"])
            col_c.metric("Collection", doc.get("collection", "-"))
            st.caption(
                f"来源：{doc['source_path']} | "
                f"Hash: `{doc['source_hash'][:16]}...` | "
                f"处理时间：{doc.get('processed_at', '-')}"
            )

            st.divider()

            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(f"### {CHUNKS} ({len(chunks)})")
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]
                    with st.container(border=True):
                        st.markdown(
                            f"**Chunk {cidx + 1}** | `{chunk_id[-16:]}` | {len(text)} 字符"
                        )
                        height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            "内容",
                            value=text,
                            height=height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        with st.expander(METADATA, expanded=False):
                            st.json(meta)
            else:
                st.caption("这个文档在向量存储中没有找到任何 Chunk。")

            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(f"### {IMAGES} ({len(images)})")
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(f"{img['image_id']}（文件不存在）")
