import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.eval import GoldenSample
from app.services.interfaces import DocumentChunk


class SyntheticBenchmarkGenerator:
    """
    Automated Synthetic Test Dataset Generator (Ragas / TruLens style).
    Generates realistic (Question, Ground Truth Answer, Key Fact Keywords) pairs
    directly from document chunks using LLM or smart heuristic extraction.
    """

    @classmethod
    def _extract_keywords(cls, text: str, max_keywords: int = 5) -> List[str]:
        """Extract high-signal words (nouns/entities, ignoring stopwords)."""
        stopwords = {
            "the", "and", "is", "in", "it", "of", "to", "for", "with", "on", "that",
            "this", "by", "from", "are", "as", "an", "be", "was", "were", "or", "at",
            "which", "also", "into", "their", "will", "can", "has", "have", "more"
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        candidates = [w for w in words if w not in stopwords]

        # Calculate word frequency
        freq = {}
        for w in candidates:
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.keys(), key=lambda w: freq[w], reverse=True)
        return sorted_words[:max_keywords]

    @classmethod
    def _heuristic_generate_sample(cls, chunk: DocumentChunk, sample_idx: int) -> Optional[GoldenSample]:
        """
        Extracts key factual statements and frames a relevant question
        without requiring an external API key (works 100% offline).
        """
        text = chunk.text.strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]

        if not sentences:
            return None

        # Pick a salient sentence
        target_sentence = sentences[0]
        for s in sentences:
            if any(term in s.lower() for term in ["is", "provides", "defines", "allows", "uses", "features", "because"]):
                target_sentence = s
                break

        keywords = cls._extract_keywords(text, max_keywords=5)
        top_keyword = keywords[0] if keywords else "the topic"

        # Frame natural question styles
        if len(keywords) >= 2:
            question_templates = [
                f"How does the document describe {keywords[0]} in relation to {keywords[1]}?",
                f"What role does {keywords[0]} play according to the text?",
                f"Explain how {keywords[0]} works and what it provides.",
                f"What are the key points regarding {keywords[0]}?",
            ]
        else:
            question_templates = [
                f"What information is provided regarding {top_keyword}?",
                f"Explain the primary purpose of {top_keyword}.",
            ]

        selected_query = question_templates[sample_idx % len(question_templates)]

        return GoldenSample(
            sample_id=f"synth_{chunk.doc_id}_{sample_idx}",
            query=selected_query,
            ground_truth_doc_ids=[chunk.doc_id],
            ground_truth_answer=target_sentence,
            relevant_keywords=keywords,
        )

    @classmethod
    async def _llm_generate_sample(
        cls, chunk: DocumentChunk, sample_idx: int, api_key: str
    ) -> Optional[GoldenSample]:
        """Generate high-quality question-answer benchmark pair using an LLM API."""
        try:
            import httpx
            prompt = (
                "You are an expert evaluator generating evaluation benchmarks for a RAG system.\n"
                "Given the context below, generate a JSON object with:\n"
                "1. 'query': A natural, specific user question that can be answered ONLY using this text.\n"
                "2. 'ground_truth_answer': A concise factual answer.\n"
                "3. 'relevant_keywords': A list of 3-5 critical keywords from the text.\n\n"
                f"CONTEXT:\n{chunk.text}\n\n"
                "Respond ONLY with valid JSON in format: "
                "{\"query\": \"...\", \"ground_truth_answer\": \"...\", \"relevant_keywords\": [\"...\"]}"
            )

            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "response_format": {"type": "json_object"},
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    parsed = json.loads(data["choices"][0]["message"]["content"])
                    return GoldenSample(
                        sample_id=f"synth_llm_{chunk.doc_id}_{sample_idx}",
                        query=parsed.get("query"),
                        ground_truth_doc_ids=[chunk.doc_id],
                        ground_truth_answer=parsed.get("ground_truth_answer"),
                        relevant_keywords=parsed.get("relevant_keywords", []),
                    )
        except Exception as e:
            logger.warning(f"LLM synthetic generation failed ({e}), using heuristic generator.")

        return cls._heuristic_generate_sample(chunk, sample_idx)

    @classmethod
    async def generate_from_chunks(
        cls,
        chunks: List[DocumentChunk],
        num_samples: int = 5,
        use_llm: bool = False,
    ) -> List[GoldenSample]:
        """
        Generates synthetic golden test benchmark samples from a list of document chunks.
        """
        if not chunks:
            return []

        samples: List[GoldenSample] = []
        api_key = settings.OPENAI_API_KEY if use_llm else None

        # Sample evenly across chunks
        step = max(1, len(chunks) // num_samples)
        selected_chunks = [chunks[i] for i in range(0, len(chunks), step)][:num_samples]

        for idx, chunk in enumerate(selected_chunks, start=1):
            if api_key and use_llm:
                sample = await cls._llm_generate_sample(chunk, idx, api_key)
            else:
                sample = cls._heuristic_generate_sample(chunk, idx)

            if sample:
                samples.append(sample)

        return samples


synthetic_generator = SyntheticBenchmarkGenerator()
