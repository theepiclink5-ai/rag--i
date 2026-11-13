# Modular RAG Cloud - 企业内部知识库系统

> 基于模块化 RAG（检索增强生成）架构的企业级智能知识库系统，通过 MCP（Model Context Protocol）协议对外暴露工具接口，支持对接 Copilot / Claude 等 AI 助手，为企业提供精准、可追溯的知识问答能力。

---

## 📖 目录

- [项目概述](#-项目概述)
- [核心能力](#-核心能力)
- [技术架构](#-技术架构)
- [快速开始](#-快速开始)
- [MCP 集成](#-mcp-集成)
- [配置说明](#-配置说明)

---

## 🏗️ 项目概述

Modular RAG Cloud 是一套面向企业内部知识管理的智能问答系统。系统将文档摄取、混合检索、重排序、LLM 生成及全链路可观测性整合为完整的工程化解决方案，帮助企业将分散的文档资料转化为可即时查询、可追溯来源的智能知识服务。

**核心价值：**

-   **精准检索**：混合检索（Dense + Sparse）+ Rerank 两段式架构，兼顾语义理解与精确匹配
-   **多模态支持**：自动识别 PDF 中的图片并生成文字描述，实现图文统一检索
-   **全链路可观测**：摄取与查询两条链路的每个中间状态透明可见，支持问题定位与效果调优
-   **标准化接口**：遵循 MCP 协议，一次开发即可对接多种 AI 客户端
-   **可插拔架构**：LLM / Embedding / Reranker / VectorStore 等核心组件均支持配置化替换，零代码修改适配不同业务场景

---

## 🔧 核心能力

| 模块 | 能力 | 说明 |
|------|------|------|
| **Ingestion Pipeline** | PDF → Markdown → Chunk → Transform → Embedding → Upsert | 全链路数据摄取，支持多模态图片描述（Image Captioning） |
| **Hybrid Search** | Dense (向量) + Sparse (BM25) + RRF Fusion + Rerank | 粗排召回 + 精排重排的两段式检索架构 |
| **MCP Server** | 标准 MCP 协议暴露 Tools | `query_knowledge_hub`、`list_collections`、`get_document_summary` |
| **Dashboard** | Streamlit 六页面管理平台 | 系统总览 / 数据浏览 / Ingestion 管理 / 摄取追踪 / 查询追踪 / 评估面板 |
| **Evaluation** | Ragas + Custom 评估体系 | 支持 golden test set 回归测试，量化评估检索与生成质量 |
| **Observability** | 全链路白盒化追踪 | Ingestion 与 Query 两条链路的每一个中间状态透明可见 |

---

## 🏛️ 技术架构

### 🔌 全链路可插拔架构

LLM / Embedding / Reranker / Splitter / VectorStore / Evaluator 每一个核心环节均定义了抽象接口，支持"乐高积木式"替换，通过配置文件一键切换后端，零代码修改。

### 🔍 混合检索 + 重排

BM25 稀疏检索解决专有名词精确匹配 + Dense Embedding 解决同义词语义匹配，RRF 融合后可选 Cross-Encoder / LLM Rerank 精排，平衡查全率与查准率。

### 🖼️ 多模态图像处理

采用 Image-to-Text 策略，利用 Vision LLM 自动生成图片描述并缝合进 Chunk，复用纯文本 RAG 链路即可实现"搜文字出图"。

### 📡 MCP 生态集成

遵循 Model Context Protocol 标准，可直接对接 GitHub Copilot、Claude Desktop 等 MCP Client，零前端开发，一次开发处处可用。

### 📊 可视化管理 + 自动化评估

Streamlit Dashboard 提供完整的数据管理与链路追踪能力，集成 Ragas 等评估框架，建立基于数据的迭代反馈回路。

### 🧪 三层测试体系

Unit / Integration / E2E 分层测试，覆盖独立模块逻辑、模块间交互、完整链路（MCP Client / Dashboard）。

> 📖 详细架构设计、模块说明请参阅 [DEV_SPEC.md](DEV_SPEC.md)

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd modular-rag-cloud
```

### 2. 环境配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 填写必要配置（API Key、数据库连接等）
vim .env

# 安装依赖（推荐使用虚拟环境）
pip install -e .
```

### 3. 启动服务

```bash
# 启动 MCP Server
python main.py

# 或启动 Dashboard
streamlit run src/observability/dashboard/app.py
```

---

## 🔗 MCP 集成

本系统通过 MCP 协议暴露以下工具接口，可供 AI 助手直接调用：

| Tool | 功能 |
|------|------|
| `query_knowledge_hub` | 对知识库进行自然语言查询，返回带引用来源的回答 |
| `list_collections` | 列出所有可用的知识集合 |
| `get_document_summary` | 获取指定文档的摘要信息 |

**支持的 MCP Client：**

-   GitHub Copilot
-   Claude Desktop
-   Cursor
-   其他兼容 MCP 协议的客户端

---

## ⚙️ 配置说明

### 切换 LLM / Embedding Provider

编辑 `config/settings.yaml`，通过配置切换不同的模型后端：

```yaml
llm:
  provider: deepseek    # 可选: openai, qwen, ollama, deepseek
  model: deepseek-chat

embedding:
  provider: jina        # 可选: openai, jina, local
  model: jina-embeddings-v3
```

### 支持的文件格式

当前 Ingestion Pipeline 原生支持 PDF 格式。如需扩展 Word / Markdown / HTML 等格式，可在 `src/libs/loader/` 下添加对应的 Loader 实现。

---

## 📄 License

本项目基于原开源项目 [MODULAR-RAG-MCP-SERVER](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER) 改进而来。
