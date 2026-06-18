"""RAG answer generator using LangChain + OpenAI-compatible LLM."""

import time
from typing import List, Optional

from openai import OpenAI

from src.core.generator import BaseGenerator
from src.core.schemas import SearchResult, GeneratedAnswer
from src.registry import ModuleRegistry
from src.generation.prompt_templates import SYSTEM_PROMPT_ZH, format_context
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.generators.register("rag")
class RAGGenerator(BaseGenerator):
    """Generate answers using an OpenAI-compatible LLM with RAG context."""

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "sk-placeholder",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_context_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT_ZH

        self._client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        logger.info(f"RAGGenerator initialized with model={model}")

    def generate(
        self,
        query: str,
        context_chunks: List[SearchResult],
    ) -> GeneratedAnswer:
        start = time.perf_counter()

        # Format context
        context = format_context(context_chunks, self.max_context_tokens)

        # Build prompt
        prompt = self.system_prompt.format(context=context, question=query)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=2048,
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"[生成回答时出错: {e}]"

        elapsed = (time.perf_counter() - start) * 1000

        # Build sources list
        sources = []
        for chunk in context_chunks[:5]:
            sources.append({
                "chunk_id": chunk.chunk_id,
                "content_snippet": chunk.content[:200],
                "doc_name": chunk.metadata.get("file_name", "Unknown"),
                "score": round(chunk.score, 4),
            })

        logger.info(
            f"Generated answer in {elapsed:.0f}ms, "
            f"{len(context_chunks)} context chunks"
        )
        return GeneratedAnswer(
            answer=answer,
            sources=sources,
            confidence=1.0,
            latency_ms=elapsed,
        )
