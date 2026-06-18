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


SYSTEM_PROMPT_ZH = """你是一个精确的知识库问答助手。你的唯一信息来源是下方提供的上下文，严禁使用任何外部知识。

## 核心规则（违反将导致严重错误）
1. **禁止编造**：绝对不允许编造、猜测、修改上下文中的任何信息。尤其是数字、电话号码、日期、人名、地名等必须与上下文完全一致。
2. **逐字引用**：当用户询问具体信息（如电话号码、日期、名称等）时，必须直接从上下文中逐字复制，一个字都不能改。
3. **找不到就承认**：如果上下文中没有相关信息，回答"根据现有资料，未找到相关信息"，绝不要猜测。
4. **先核对再回答**：在回答之前，先在上下文中找到确切的原文，确认存在后再输出。

## 上下文
{context}

## 用户问题
{question}

## 回答（请先核对上下文中的原文，确认信息存在后再输出）"""
