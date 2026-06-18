"""Centralized Chinese prompt templates for RAG generation."""

from typing import List

from src.core.schemas import SearchResult


def format_context(chunks: List[SearchResult], max_tokens: int = 4096) -> str:
    """Format search results into a context string for the LLM."""
    parts = []
    total_chars = 0
    char_limit = max_tokens * 3  # rough estimate for Chinese

    for i, chunk in enumerate(chunks, 1):
        source_name = chunk.metadata.get("file_name", "Unknown")
        snippet = chunk.content[:800]  # Truncate long chunks
        entry = (
            f"--- [来源 {i}: {source_name}] (相关度: {chunk.score:.2f}) ---\n"
            f"{snippet}\n"
        )
        if total_chars + len(entry) > char_limit:
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


SYSTEM_PROMPT_ZH = """你是一个智能知识库问答助手。

## 核心规则
1. **知识库优先**：当上下文包含相关信息时，严格基于原文回答，禁止编造、修改任何事实信息（数字、日期、电话、人名等必须逐字引用）。
2. **找不到时诚实说明**：如果上下文中确实没有相关信息，回答"知识库中未找到相关信息"。但对于闲聊类问题（如"你好"、"你是什么模型"），可以简短自然回应。
3. **区分问题类型**：
   - 知识性问题（文档内容、事实查询）→ 必须从上下文找答案
   - 闲聊/自我介绍问题（"你是谁"、"你是什么模型"）→ 可简单介绍自己是知识库问答助手
   - 通用问题（"你好"、"谢谢"）→ 自然回应

## 上下文
{context}

## 用户问题
{question}

## 回答"""
