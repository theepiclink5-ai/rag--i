"""Modular RAG Dashboard - multi-page Streamlit application."""
from __future__ import annotations
import sys

# Patch sqlite3 for ChromaDB compatibility (requires >= 3.35.0)
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st



def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


pages = [
    st.Page(_page_overview, title="系统概览", icon="📊", default=True),
    st.Page(_page_data_browser, title="数据浏览", icon="🔍"),
    st.Page(_page_ingestion_manager, title="导入管理", icon="📥"),
    st.Page(_page_ingestion_traces, title="导入追踪", icon="🔬"),
    st.Page(_page_query_traces, title="查询追踪", icon="🔎"),
    st.Page(_page_evaluation_panel, title="评估面板", icon="📏"),
]


def main() -> None:
    st.set_page_config(
        page_title="模块化 RAG 控制台",
        page_icon="📊",
        layout="wide",
    )

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    main()
