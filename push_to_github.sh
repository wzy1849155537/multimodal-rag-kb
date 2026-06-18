#!/bin/bash
# Push multimodal-rag-kb to GitHub
cd "d:/edgedownload/Meta-Memory-main/multimodal-rag-kb"

git add -A
git commit -m "feat: 多模态RAG知识库问答系统

- 离线索引: PDF/MD/TXT解析, 递归切分, Chunk清洗, 混合索引(BM25+ChromaDB)
- 在线检索: Query改写, 多路召回(BM25+稠密), RRF融合, Cross-Encoder重排
- 扫描件OCR: VLM(Qwen3-VL-8B)自动提取图片PDF文字
- 增量索引: SHA256哈希变更检测, 热更新同步
- 测评: RAGAS + MRR + Hit@k + 网格搜索实验
- 桌面版: PyWebView原生Windows窗口
- 工厂模式: 所有模块可插拔配置"

git branch -M main
git push -u origin main
echo "Done!"
