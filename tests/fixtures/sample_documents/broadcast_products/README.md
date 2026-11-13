# 广播产品资料库 RAG 演示资料集

这是一套虚构的中型广播产品资料库，用来展示多文件导入、长文档切片、Hybrid Search、Rerank、多文档组合推荐、引用溯源和 Evaluation。

- products/speakers/: 20 份音箱规格书
- products/amplifiers/: 12 份功放规格书
- products/microphones/: 10 份话筒规格书
- certifications/: 6 份检测报告台账
- business/: 5 份业务规则、价格、库存、售后资料
- guides/: 7 份场景选型和安装指南

共 60 份业务 Markdown 文档，不含本 README。

导入命令：`python scripts\ingest.py --path tests\fixtures\sample_documents\broadcast_products --collection broadcast_products --force`

核心问题：我们要给一个 800 平米的室外操场做广播，要求可售、报告有效、音箱防水、功放至少 4 分区、需要分区寻呼话筒，请推荐组合。预期推荐：BC-OUT-450P + BC-AMP-500Z + BC-MIC-910C。
