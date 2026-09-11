# app/services/generation_service.py
from app.core.llm_client import LLMClient
from app.core.prompts import build_rag_prompt


class GenerationService:
    """
    Orchestrates prompt building + LLM generation.
    """
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def generate(self, query: str, context_chunks: list[dict]) -> str:
        prompt = build_rag_prompt(context_chunks, query)
        return await self.llm.generate(prompt)

    async def generate_stream(self, query: str, context_chunks: list[dict]):
        prompt = build_rag_prompt(context_chunks, query)
        async for token in self.llm.generate_stream(prompt):
            yield token