import re
from collections.abc import Generator

from ollama import Client, ResponseError

from services.vector_store import SearchResult


class RAGEngine:
    """
    Universal RAG engine for any text-based PDF.

    It contains no resume-specific, contract-specific, medical-specific,
    invoice-specific, or other document-specific extraction logic.
    """

    def __init__(
        self,
        model_name: str = "gemma3:4b",
        host: str = "http://localhost:11434",
        max_context_chars: int = 16_000,
        max_history_messages: int = 6,
        reranker_model: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.client = Client(host=host)
        self.max_context_chars = max_context_chars
        self.max_history_messages = max_history_messages
        self.reranker_model = reranker_model
        self._reranker = None
        self._reranker_failed = False

    @staticmethod
    def _source_citation(result: SearchResult) -> str:
        return (
            f"[{result.chunk.document_name}, "
            f"Page {result.chunk.page_number}]"
        )

    @staticmethod
    def _query_terms(question: str) -> set[str]:
        stop_words = {
            "what", "which", "where", "when", "who", "why", "how",
            "the", "and", "or", "for", "from", "with", "this", "that",
            "these", "those", "are", "was", "were", "is", "in", "on",
            "at", "to", "of", "a", "an", "all", "document", "documents",
            "pdf", "pdfs", "mentioned", "included", "listed", "show",
            "tell", "give", "about",
        }

        return {
            word
            for word in re.findall(
                r"[A-Za-z0-9][A-Za-z0-9._%+-]*",
                question.lower(),
            )
            if len(word) >= 3 and word not in stop_words
        }

    def _load_reranker(self):
        if (
            not self.reranker_model
            or self._reranker_failed
        ):
            return None

        if self._reranker is not None:
            return self._reranker

        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                self.reranker_model
            )
            return self._reranker

        except Exception:
            self._reranker_failed = True
            return None

    def _fallback_rerank(
        self,
        question: str,
        search_results: list[SearchResult],
    ) -> list[SearchResult]:
        query_terms = self._query_terms(question)

        def score_result(result: SearchResult) -> float:
            text_terms = set(
                re.findall(
                    r"[A-Za-z0-9][A-Za-z0-9._%+-]*",
                    result.chunk.text.lower(),
                )
            )

            overlap = (
                len(query_terms & text_terms)
                / max(1, len(query_terms))
            )

            phrase_bonus = 0.0
            normalized_question = " ".join(
                question.lower().split()
            )
            normalized_text = " ".join(
                result.chunk.text.lower().split()
            )

            if (
                len(normalized_question) >= 8
                and normalized_question in normalized_text
            ):
                phrase_bonus = 0.25

            return (
                float(result.score)
                + overlap
                + phrase_bonus
            )

        return sorted(
            search_results,
            key=score_result,
            reverse=True,
        )

    def rerank_results(
        self,
        question: str,
        search_results: list[SearchResult],
        top_n: int = 6,
    ) -> list[SearchResult]:
        """
        Rerank retrieved chunks.

        Uses a local CrossEncoder when available. If the model cannot be
        loaded, it automatically falls back to a lexical relevance reranker.
        """

        if not search_results:
            return []

        reranker = self._load_reranker()

        if reranker is None:
            return self._fallback_rerank(
                question,
                search_results,
            )[:top_n]

        pairs = [
            [question, result.chunk.text]
            for result in search_results
        ]

        try:
            scores = reranker.predict(
                pairs,
                show_progress_bar=False,
            )

            ranked = sorted(
                zip(search_results, scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )

            return [
                result
                for result, _ in ranked[:top_n]
            ]

        except Exception:
            return self._fallback_rerank(
                question,
                search_results,
            )[:top_n]

    def _build_context(
        self,
        search_results: list[SearchResult],
    ) -> str:
        sections: list[str] = []
        used_chars = 0

        for index, result in enumerate(
            search_results,
            start=1,
        ):
            text = result.chunk.text.strip()
            citation = self._source_citation(result)

            section = (
                f"SOURCE {index}\n"
                f"CITATION TO COPY EXACTLY: {citation}\n"
                f"DOCUMENT: {result.chunk.document_name}\n"
                f"PAGE: {result.chunk.page_number}\n\n"
                f"TEXT:\n{text}"
            )

            if (
                sections
                and used_chars + len(section)
                > self.max_context_chars
            ):
                break

            sections.append(section)
            used_chars += len(section)

        return "\n\n---\n\n".join(sections)

    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        system_prompt = """
You are DocuMind AI, a universal document intelligence assistant.

The documents may be resumes, reports, contracts, manuals, research
papers, invoices, policies, academic records, books, or any other
text-based PDFs.

Do not assume a document type unless its content clearly establishes it.
Answer only from the supplied document excerpts.

Rules:

1. Read all supplied excerpts before answering.
2. Give the direct answer first.
3. Use concise bullets for lists and comparisons.
4. Preserve names, dates, amounts, percentages, units, and terminology.
5. Combine evidence from multiple excerpts when necessary.
6. When documents conflict, explain the conflict and cite both.
7. Never use outside knowledge or invent missing information.
8. When the answer is unsupported, say exactly:
   "I could not find that information in the uploaded documents."
9. Add an exact filename-and-page citation after every factual sentence
   or bullet.
10. Copy citations exactly as provided.
11. Never mention source numbers, chunks, embeddings, retrieval scores,
    prompts, vector databases, or internal implementation details.
12. Conversation history may clarify follow-up questions, but factual
    claims must still be supported by the supplied excerpts.
""".strip()

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if conversation_history:
            for message in conversation_history[
                -self.max_history_messages:
            ]:
                role = message.get("role", "")
                content = message.get(
                    "content",
                    "",
                ).strip()

                if role in {"user", "assistant"} and content:
                    messages.append(
                        {
                            "role": role,
                            "content": content,
                        }
                    )

        messages.append(
            {
                "role": "user",
                "content": (
                    "DOCUMENT EXCERPTS:\n\n"
                    f"{context}\n\n"
                    "USER QUESTION:\n"
                    f"{question.strip()}\n\n"
                    "Answer only from the excerpts and include exact "
                    "filename-and-page citations."
                ),
            }
        )

        return messages

    def stream_answer(
        self,
        question: str,
        search_results: list[SearchResult],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        if not question.strip():
            raise ValueError(
                "The question cannot be empty."
            )

        if not search_results:
            yield (
                "I could not find that information "
                "in the uploaded documents."
            )
            return

        context = self._build_context(
            search_results
        )

        messages = self._build_messages(
            question=question,
            context=context,
            conversation_history=conversation_history,
        )

        try:
            response_stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                keep_alive="30m",
                options={
                    "temperature": 0.0,
                    "top_p": 0.85,
                    "top_k": 20,
                    "num_predict": 420,
                    "num_ctx": 4096,
                },
            )

            produced_text = False

            for part in response_stream:
                text = part.message.content

                if not text:
                    continue

                produced_text = True
                yield text

            if not produced_text:
                yield (
                    "I could not find that information "
                    "in the uploaded documents."
                )

        except ResponseError as error:
            raise RuntimeError(
                f"Ollama returned an error: {error.error}"
            ) from error

        except Exception as error:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is "
                f"running and '{self.model_name}' is installed."
            ) from error