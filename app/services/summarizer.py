from app.services.llm import LLMService
from app.models import Paper
from typing import List

MAP_SYSTEM = 'You are summarizing a section of a research paper. Be concise and retain key technical details.'
REDUCE_SYSTEM = """You are producing a final structured summary of a research paper.
Use the section summaries provided. Structure your output as:
OBJECTIVE: [1-2 sentences]
METHODOLOGY: [2-3 sentences]
KEY FINDINGS: [3-5 bullet points]
LIMITATIONS: [1-2 sentences]
CONCLUSION: [1-2 sentences]"""

class SummarizerService:
    def __init__(self, provider: str = None, model: str = None):
        self.llm = LLMService(provider=provider, model=model)

    async def _summarize_chunk(self, chunk_text: str, paper_title: str) -> str:
        prompt = f'Paper: {paper_title}\n\nSection:\n{chunk_text}\n\nSummarize this section:'
        return await self.llm.complete(MAP_SYSTEM, prompt, max_tokens=300)

    async def summarize_paper(self, paper: Paper, chunks: List[dict]) -> str:
        if not chunks:
            return "No content available to summarize."
            
        import asyncio
            
        # MAP: summarize key chunks (cap at 3 to stay within free-tier rate limits of 5 RPM)
        max_map_chunks = 3
        step = max(1, len(chunks) // max_map_chunks)
        sampled = chunks[::step][:max_map_chunks]

        chunk_summaries = []
        for i, c in enumerate(sampled):
            if i > 0:
                await asyncio.sleep(10.0)  # sleep 10 seconds between requests to avoid rate limits
            summary = await self._summarize_chunk(c['text'], paper.title)
            chunk_summaries.append(summary)

        # Wait 10 seconds before the final reduce call to avoid rate limits
        await asyncio.sleep(10.0)

        # REDUCE: combine all summaries into final
        combined = '\n\n---\n\n'.join(chunk_summaries)
        reduce_prompt = f"""Paper Title: {paper.title}
Authors: {paper.authors or 'Unknown'}
Year: {paper.year or 'n.d.'}

Section summaries:
{combined}

Produce the final structured summary:"""

        return await self.llm.complete(REDUCE_SYSTEM, reduce_prompt, max_tokens=800)
