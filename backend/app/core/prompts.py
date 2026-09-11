# app/core/prompts.py
"""
Prompt templates for the RAG pipeline.

Why a dedicated module:
- Prompts are the single most impactful lever on answer quality.
- Keeping them isolated makes A/B testing trivial.
- Centralizing them means every service uses the same grounding rules.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompt — enforces grounding, citation, and abstention
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a helpful assistant answering questions based ONLY on the provided context.

Rules:
1. If the context does not contain enough information to answer, respond exactly:
   "I don't have enough information to answer this question."
2. Do NOT use prior knowledge. Only use the context below.
3. Cite the source and page when possible using the format [n] where n is
   the chunk number in the context.
4. Be concise, accurate, and never fabricate details.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------
def format_context(chunks: list[dict]) -> str:
    """
    Render retrieved chunks as numbered context blocks.

    Each block carries a numeric marker and source metadata, so the LLM can
    cite precisely. Numbered markers also make it easy to trace which chunk
    an answer used.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[{i}] (Source: {source}, Page: {page})\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry point used by generation_service.py
# ---------------------------------------------------------------------------
def build_rag_prompt(context_chunks: list[dict], question: str) -> str:
    """
    Build the final prompt string sent to the LLM.

    Args:
        context_chunks: retrieved + reranked chunks (dicts with text, source, page)
        question:       the user's question

    Returns:
        A fully-formed prompt ready to pass to `LLMClient.generate(...)`.
    """
    context = format_context(context_chunks)
    return RAG_SYSTEM_PROMPT.format(context=context, question=question)