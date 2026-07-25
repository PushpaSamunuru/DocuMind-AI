import re
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from services.text_splitter import TextChunk


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float


class VectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[TextChunk] = []
        self.embeddings: np.ndarray | None = None

    def build(
        self,
        chunks: list[TextChunk],
    ) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build the vector store without chunks."
            )

        self.chunks = chunks

        texts = [
            f"Document: {chunk.document_name}\n{chunk.text}"
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
            "who",
            "when",
            "where",
            "how",
            "my",
            "me",
            "i",
            "do",
            "did",
            "does",
            "have",
            "has",
            "from",
            "in",
            "on",
            "of",
            "to",
            "and",
            "or",
            "for",
            "with",
            "all",
            "shown",
            "listed",
            "mentioned",
            "please",
        }

        return {
            word
            for word in words
            if word not in stop_words
        }

    @staticmethod
    def _expand_query(query: str) -> str:
        query_lower = query.lower()
        additions: list[str] = []

        if "gpa" in query_lower:
            additions.extend(
                [
                    "GPA",
                    "grade point average",
                    "cumulative GPA",
                    "academic record",
                    "grades",
                ]
            )

        if any(
            word in query_lower
            for word in [
                "university",
                "college",
                "education",
                "degree",
                "graduated",
                "graduation",
            ]
        ):
            additions.extend(
                [
                    "Education",
                    "university",
                    "college",
                    "Master of Science",
                    "Bachelor of Technology",
                    "degree",
                    "graduation",
                ]
            )

        if any(
            word in query_lower
            for word in [
                "project",
                "projects",
                "portfolio",
            ]
        ):
            additions.extend(
                [
                    "Projects",
                    "built",
                    "developed",
                    "created",
                    "application",
                    "system",
                    "agent",
                ]
            )

        if any(
            word in query_lower
            for word in [
                "experience",
                "employment",
                "professional",
                "work",
            ]
        ):
            additions.extend(
                [
                    "Experience",
                    "professional experience",
                    "employment",
                    "internship",
                    "engineer",
                    "developer",
                    "analyst",
                ]
            )

        if "skill" in query_lower:
            additions.extend(
                [
                    "Skills",
                    "technical skills",
                    "programming languages",
                    "frameworks",
                    "libraries",
                    "tools",
                ]
            )

        if "date" in query_lower:
            additions.extend(
                [
                    "date",
                    "dated",
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ]
            )

        return f"{query} {' '.join(additions)}".strip()

    @staticmethod
    def _filename_boost(
        query: str,
        filename: str,
    ) -> float:
        query_lower = query.lower()
        filename_lower = filename.lower()

        boost = 0.0

        specifically_resume = (
            "resume" in query_lower
            or "cv" in query_lower
        )

        specifically_cover_letter = (
            "cover letter" in query_lower
            or "coverletter" in query_lower
        )

        if specifically_resume:
            if "resume" in filename_lower:
                boost += 3.0

            if "cover" in filename_lower:
                boost -= 2.0

        if specifically_cover_letter:
            if "cover" in filename_lower:
                boost += 3.0

            if "resume" in filename_lower:
                boost -= 1.0

        if any(
            word in query_lower
            for word in ["gpa", "grade", "grades"]
        ):
            if any(
                word in filename_lower
                for word in [
                    "grade",
                    "transcript",
                    "academic",
                ]
            ):
                boost += 2.0

            if "resume" in filename_lower:
                boost += 1.2

            if "cover" in filename_lower:
                boost += 0.4

        if any(
            word in query_lower
            for word in [
                "university",
                "education",
                "degree",
                "graduated",
                "graduation",
            ]
        ):
            if any(
                word in filename_lower
                for word in [
                    "resume",
                    "grade",
                    "transcript",
                    "academic",
                ]
            ):
                boost += 1.5

        if any(
            word in query_lower
            for word in [
                "project",
                "projects",
                "portfolio",
            ]
        ):
            if "resume" in filename_lower:
                boost += 2.5

            if "cover" in filename_lower:
                boost += 0.3

        if any(
            word in query_lower
            for word in [
                "skill",
                "skills",
                "experience",
                "professional",
            ]
        ):
            if "resume" in filename_lower:
                boost += 1.8

        if "date" in query_lower:
            if "cover" in filename_lower:
                boost += 1.8

        if "compare" in query_lower:
            if any(
                word in filename_lower
                for word in ["resume", "cover"]
            ):
                boost += 1.5

        return boost

    @staticmethod
    def _pattern_boost(
        query: str,
        text: str,
    ) -> float:
        query_lower = query.lower()
        text_lower = text.lower()

        boost = 0.0

        if "gpa" in query_lower:
            if re.search(
                r"\bgpa\s*[:\-]?\s*[0-4](?:\.\d{1,3})?\b",
                text_lower,
            ):
                boost += 3.0

            elif "gpa" in text_lower:
                boost += 1.5

        if any(
            word in query_lower
            for word in [
                "university",
                "education",
                "degree",
            ]
        ):
            if "university" in text_lower:
                boost += 1.5

            if re.search(
                r"\b(master|bachelor|doctor|phd)\b",
                text_lower,
            ):
                boost += 1.2

            if "education" in text_lower:
                boost += 1.0

        if any(
            word in query_lower
            for word in ["project", "projects"]
        ):
            if "projects" in text_lower:
                boost += 2.5

            project_phrases = [
                "gesture-controlled",
                "emotion detection",
                "health insights",
                "rock-paper-scissors",
                "built",
                "developed",
                "created",
            ]

            boost += sum(
                0.35
                for phrase in project_phrases
                if phrase in text_lower
            )

        if "date" in query_lower:
            date_patterns = [
                r"\b(?:january|february|march|april|may|june|"
                r"july|august|september|october|november|december)"
                r"\s+\d{1,2},\s+\d{4}\b",
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b\d{4}-\d{2}-\d{2}\b",
            ]

            if any(
                re.search(pattern, text_lower)
                for pattern in date_patterns
            ):
                boost += 2.0

        return boost

    def search(
        self,
        query: str,
        top_k: int = 6,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        if self.index is None or not self.chunks:
            raise ValueError(
                "The vector store has not been built."
            )

        expanded_query = self._expand_query(query)

        query_embedding = self.model.encode(
            [expanded_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        semantic_scores, semantic_indices = self.index.search(
            query_embedding,
            len(self.chunks),
        )

        query_tokens = self._tokenize(expanded_query)

        scored_results: list[SearchResult] = []

        for semantic_score, chunk_index in zip(
            semantic_scores[0],
            semantic_indices[0],
        ):
            if chunk_index < 0:
                continue

            chunk = self.chunks[int(chunk_index)]

            combined_text = (
                f"{chunk.document_name} {chunk.text}"
            )

            chunk_tokens = self._tokenize(combined_text)

            overlap_count = len(
                query_tokens.intersection(chunk_tokens)
            )

            keyword_score = (
                overlap_count
                / max(len(query_tokens), 1)
            )

            filename_boost = self._filename_boost(
                query=query,
                filename=chunk.document_name,
            )

            pattern_boost = self._pattern_boost(
                query=query,
                text=chunk.text,
            )

            combined_score = (
                float(semantic_score) * 1.0
                + keyword_score * 2.0
                + filename_boost
                + pattern_boost
            )

            scored_results.append(
                SearchResult(
                    chunk=chunk,
                    score=combined_score,
                )
            )

        scored_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        unique_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for result in scored_results:
            chunk_id = getattr(
                result.chunk,
                "chunk_id",
                (
                    f"{result.chunk.document_name}-"
                    f"{result.chunk.page_number}-"
                    f"{result.chunk.text[:100]}"
                ),
            )

            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)
            unique_results.append(result)

            if len(unique_results) >= top_k:
                break

        return unique_results