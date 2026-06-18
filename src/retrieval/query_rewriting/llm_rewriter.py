"""LLM-based query rewriter for expansion and decomposition."""

from typing import List

from openai import OpenAI

from src.core.query_rewriter import BaseQueryRewriter
from src.core.schemas import RewrittenQuery
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

REWRITE_PROMPT = """你是一个查询改写助手。请将用户的原始问题改写为更适合检索的多个版本。

## 规则
1. 保持原意，不添加新信息
2. 生成 {max_rewrites} 个改写版本，每个版本从不同角度表达
3. 可以：展开缩写、补充同义词、分解复合问题、调整语序
4. 输出格式：每行一个改写版本

## 原始问题
{query}

## 改写版本（每行一个）"""


@ModuleRegistry.rewriters.register("llm")
class LLMRewriter(BaseQueryRewriter):
    """Uses an LLM to expand and decompose queries."""

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        max_rewrites: int = 3,
    ):
        self.max_rewrites = max_rewrites
        self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._model = model

    def rewrite(self, query: str) -> List[RewrittenQuery]:
        prompt = REWRITE_PROMPT.format(query=query, max_rewrites=self.max_rewrites)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            text = response.choices[0].message.content or ""
            lines = [l.strip() for l in text.split("\n") if l.strip()]

            results = [RewrittenQuery(
                original=query, rewritten=query, rewrite_type="original"
            )]
            for line in lines[:self.max_rewrites]:
                results.append(RewrittenQuery(
                    original=query,
                    rewritten=line,
                    rewrite_type="llm_expand",
                ))
            return results
        except Exception as e:
            logger.error(f"LLM rewrite failed: {e}")
            return [RewrittenQuery(
                original=query, rewritten=query, rewrite_type="noop"
            )]
