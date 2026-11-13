# 评估面板数据集 JSON 格式规范

> **适用模块**：modular-rag-cloud Dashboard - 评估面板
> **版本**：v1.0
> **最后更新**：2025-01

---

## 1. 概述

评估面板通过上传 JSON 文件加载问答对测试集（Golden Test Set），用于驱动检索质量与生成质量的批量评估。
本文档定义了该 JSON 文件的**标准格式、字段语义、校验规则及示例**，作为数据集设计与前后端解析的唯一权威依据。

---

## 2. 顶层结构

支持以下两种等价顶层格式，解析器自动识别：

### 格式 A：带 test_cases 包装（推荐）

```json
{
  "test_cases": [ ... ]
}
```

### 格式 B：纯数组

```json
[ ... ]
```

> 注意：其他顶层类型（字符串、数字、null 等）将被拒绝并提示格式错误。

---

## 3. 测试用例字段定义

每个测试用例为一个 JSON 对象，字段定义如下：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| query | string | 是 | - | 待评估的用户查询文本，不可为空字符串 |
| reference_answer | string / null | 否 | null | 人工标注的参考答案，用于 LLM-as-Judge 类指标（faithfulness、answer_relevancy、answer_correctness、context_recall）。未提供时系统使用检索片段拼接作为替代回答 |
| expected_chunk_ids | string[] | 否 | [] | Ground-truth chunk ID 列表，用于计算检索指标（hit_rate@K、MRR），仅在 Custom/Composite evaluator 后端下生效 |
| expected_sources | string[] | 否 | [] | Ground-truth 源文件名列表，作为 expected_chunk_ids 的补充 ground truth，用于 Custom evaluator |

### 额外字段处理

- 测试用例中出现的未定义字段将被静默忽略，不会导致解析失败。
- 建议不要在数据集中添加自定义扩展字段，以免与未来版本新增字段冲突。

---

## 4. 校验规则

### 4.1 前端校验（上传时）

| 校验项 | 行为 |
|--------|------|
| 非合法 JSON | 显示 JSON 解析失败及具体错误信息 |
| 顶层非 dict/list | 提示需要包含 test_cases 数组或顶层为数组 |
| 合法且含有效 test_cases | 显示已加载 N 条测试用例 |

### 4.2 后端校验（运行时）

| 校验项 | 行为 |
|--------|------|
| query 字段缺失 | 抛出 KeyError，评估中断 |
| query 为空字符串 | 不拦截，但可能导致无意义评估结果 |
| expected_chunk_ids / expected_sources 非数组 | 不拦截，传入 evaluator 后可能报错 |
| reference_answer 为非字符串非 null | 不拦截，转为字符串使用 |

> 已知限制：当前后端对单条用例的内容校验不足，缺少 query 时直接抛异常而非友好提示。建议在后续版本中增加逐条前置校验。

---

## 5. 完整示例

### 5.1 最小可用数据集（仅 query）

```json
{
  "test_cases": [
    { "query": "什么是模块化 RAG？" },
    { "query": "如何配置 Azure OpenAI？" }
  ]
}
```

### 5.2 完整字段数据集（推荐生产使用）

```json
{
  "description": "项目知识库评估测试集 v2",
  "version": "2.0",
  "test_cases": [
    {
      "query": "什么是模块化 RAG？",
      "reference_answer": "模块化 RAG 是一种检索增强生成系统，其组件可独立配置和替换。",
      "expected_chunk_ids": ["chunk_001", "chunk_002"],
      "expected_sources": ["architecture.md"]
    },
    {
      "query": "混合检索的工作原理是什么？",
      "reference_answer": "混合检索结合稠密向量检索与 BM25 稀疏检索，通过 RRF 融合排序结果以提升召回率。",
      "expected_chunk_ids": ["chunk_015"],
      "expected_sources": ["retrieval.md"]
    },
    {
      "query": "系统支持哪些评估指标？",
      "reference_answer": "支持 hit_rate、MRR 以及 Ragas LLM-as-Judge 指标，包括 faithfulness、answer_relevancy 和 context_precision。"
    }
  ]
}
```

### 5.3 纯数组格式（等价写法）

```json
[
  {
    "query": "什么是模块化 RAG？",
    "reference_answer": "模块化 RAG 是一种检索增强生成系统。"
  }
]
```

---

## 6. Evaluator 后端与字段依赖关系

| Evaluator 后端 | query | reference_answer | expected_chunk_ids | expected_sources |
|----------------|-------|------------------|--------------------|------------------|
| ragas | 必需 | 强烈建议 | 不使用 | 不使用 |
| custom | 必需 | 可选 | 必需（否则 hit_rate/MRR 恒为 0） | 可选 |
| composite | 必需 | 建议 | 建议 | 可选 |

> 建议：若仅使用 ragas 后端，只需填写 query + reference_answer；若需评估检索命中率，必须使用 custom/composite 后端并提供 expected_chunk_ids。

---

## 7. 文件大小与性能建议

| 规模 | 预估耗时 | 建议 |
|------|----------|------|
| 1-20 条 | < 1 min | 适合快速验证 |
| 20-100 条 | 1-10 min | 适合日常回归评估 |
| 100-500 条 | 10-60 min | 适合正式发布前全量评估 |
| > 500 条 | > 1 h | 建议分批或使用离线脚本 |

---

## 8. 常见错误排查

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 上传后提示格式不正确 | 顶层不是 dict/list 或缺少 test_cases 键 | 检查顶层结构 |
| 开始评估后报 KeyError: query | 某条测试用例缺少 query 字段 | 逐条检查并补全 query |
| hit_rate / MRR 始终为 0 | 使用 ragas 后端或未填写 expected_chunk_ids | 切换至 custom/composite 后端并补充 ground truth |
| faithfulness 等指标缺失 | 未提供 reference_answer | 在数据集中补充参考答案 |
| 评估超时 | 数据集过大或 LLM 响应慢 | 减少用例数量或增大超时配置 |

---

## 9. 版本变更记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2025-01 | 初始版本，基于现有 GoldenTestCase.from_dict() 及 evaluation_panel.py 解析逻辑整理 |
