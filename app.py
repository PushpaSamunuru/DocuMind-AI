import hashlib
import html
import re
from collections.abc import Generator
from io import BytesIO
from typing import Any

import streamlit as st
import fitz

from services.pdf_processor import extract_pdf_pages
from services.rag_engine import RAGEngine
from services.text_splitter import TextChunk, create_document_chunks
from services.vector_store import SearchResult, VectorStore


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.25rem;
        padding-bottom: 5rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 16px;
        padding: 0.85rem;
        margin-bottom: 0.8rem;
        background: rgba(128, 128, 128, 0.025);
    }

    .hero-card {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 20px;
        padding: 22px 24px;
        margin: 8px 0 18px 0;
        background: linear-gradient(
            135deg,
            rgba(255, 75, 75, 0.07),
            rgba(128, 128, 128, 0.03)
        );
    }

    .document-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .document-name {
        font-weight: 700;
        word-break: break-word;
    }

    .document-details,
    .small-muted {
        opacity: 0.72;
        font-size: 0.84rem;
        margin-top: 4px;
    }

    .source-card {
        border-left: 4px solid #ff4b4b;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0 12px 0;
        background: rgba(128, 128, 128, 0.04);
    }

    .evidence-box {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 12px;
        line-height: 1.55;
        background: rgba(128, 128, 128, 0.025);
    }

    mark {
        padding: 0.08rem 0.16rem;
        border-radius: 4px;
    }

    .viewer-title {
        font-weight: 700;
        margin-bottom: 0.35rem;
        word-break: break-word;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_rag_engine() -> RAGEngine:
    return RAGEngine(
        model_name="gemma3:4b",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


@st.cache_resource(show_spinner=False)
def build_document_index(
    file_data: tuple[tuple[str, bytes], ...],
) -> tuple[
    list[Any],
    list[TextChunk],
    list[dict[str, Any]],
    VectorStore | None,
]:
    if not file_data:
        return [], [], [], None

    all_pages: list[Any] = []
    all_chunks: list[TextChunk] = []
    document_details: list[dict[str, Any]] = []

    for filename, file_bytes in file_data:
        if not file_bytes:
            continue

        pages = extract_pdf_pages(
            uploaded_file=BytesIO(file_bytes),
            document_name=filename,
        )

        chunks = create_document_chunks(pages)

        all_pages.extend(pages)
        all_chunks.extend(chunks)

        document_details.append(
            {
                "name": filename,
                "size_kb": len(file_bytes) / 1024,
                "pages": len(pages),
                "chunks": len(chunks),
                "characters": sum(len(page.text) for page in pages),
            }
        )

    if not all_chunks:
        return all_pages, [], document_details, None

    vector_store = VectorStore()
    vector_store.build(all_chunks)

    return all_pages, all_chunks, document_details, vector_store


def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "active_document_signature": "",
        "pending_question": None,
        "active_pdf_name": None,
        "active_pdf_page": 1,
        "active_highlight_query": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None


def create_file_signature(uploaded_files: list[Any]) -> str:
    signature_parts: list[str] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        signature_parts.append(
            f"{uploaded_file.name}:{len(file_bytes)}:{file_hash}"
        )

    combined_signature = "|".join(sorted(signature_parts))

    return hashlib.sha256(
        combined_signature.encode("utf-8")
    ).hexdigest()


def build_source_data(
    search_results: list[SearchResult],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, int, str]] = set()

    for result in search_results:
        chunk = result.chunk

        source_key = (
            chunk.document_name,
            chunk.page_number,
            chunk.text,
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "document": chunk.document_name,
                "page": chunk.page_number,
                "text": chunk.text,
                "score": float(result.score),
            }
        )

    return sources


def extract_query_terms(question: str) -> list[str]:
    stop_words = {
        "what", "which", "where", "when", "who", "why", "how",
        "the", "and", "or", "for", "from", "with", "this", "that",
        "these", "those", "are", "was", "were", "is", "in", "on",
        "at", "to", "of", "a", "an", "all", "document", "documents",
        "pdf", "pdfs", "mentioned", "included", "listed", "show",
        "tell", "give", "about",
    }

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9._%+-]*", question.lower())

    return [
        word
        for word in words
        if len(word) >= 3 and word not in stop_words
    ][:12]


def highlight_evidence(text: str, question: str) -> str:
    escaped_text = html.escape(text)
    terms = extract_query_terms(question)

    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(
            rf"(?i)(?<![A-Za-z0-9])({re.escape(term)})(?![A-Za-z0-9])"
        )
        escaped_text = pattern.sub(r"<mark>\1</mark>", escaped_text)

    return escaped_text.replace("\n", "<br>")


def open_source(
    document: str,
    page: int,
    question: str,
) -> None:
    st.session_state.active_pdf_name = document
    st.session_state.active_pdf_page = max(1, int(page))
    st.session_state.active_highlight_query = question


def display_sources(
    sources: list[dict[str, Any]],
    question: str,
    key_prefix: str,
) -> None:
    if not sources:
        return

    with st.expander(
        f"Evidence and sources ({len(sources)})",
        expanded=False,
    ):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"""
<div class="source-card">
    <strong>📄 {html.escape(source["document"])}</strong><br>
    <span class="small-muted">
        Page {source["page"]} · retrieval score {source["score"]:.3f}
    </span>
</div>
""",
                unsafe_allow_html=True,
            )

            if st.button(
                f"Open {source['document']} on page {source['page']}",
                key=f"{key_prefix}_source_{number}",
                use_container_width=True,
            ):
                open_source(
                    source["document"],
                    source["page"],
                    question,
                )
                st.rerun()

            highlighted = highlight_evidence(
                source["text"],
                question,
            )

            st.markdown(
                f'<div class="evidence-box">{highlighted}</div>',
                unsafe_allow_html=True,
            )

            if number < len(sources):
                st.divider()


def display_chat_history() -> None:
    for index, message in enumerate(st.session_state.messages):
        avatar = "👤" if message["role"] == "user" else "🤖"

        with st.chat_message(
            message["role"],
            avatar=avatar,
        ):
            st.markdown(message["content"])

            if message.get("sources"):
                display_sources(
                    message["sources"],
                    message.get("question", ""),
                    key_prefix=f"history_{index}",
                )


def stream_to_screen(
    token_stream: Generator[str, None, None],
) -> Generator[str, None, None]:
    yield from token_stream


def get_top_k(question: str, total_chunks: int) -> int:
    question_lower = question.lower()

    broad_question_terms = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "similarities",
        "summarize all",
        "summarise all",
        "summary of all",
        "across all documents",
        "both documents",
        "all documents",
        "all pdfs",
        "overall",
        "main themes",
        "recommendations",
        "conclusions",
    ]

    requested = (
        12
        if any(term in question_lower for term in broad_question_terms)
        else 8
    )

    return min(requested, total_chunks)


def get_conversation_history() -> list[dict[str, str]]:
    history: list[dict[str, str]] = []

    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content", "")

        if (
            role in {"user", "assistant"}
            and isinstance(content, str)
            and content.strip()
        ):
            history.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    return history


def render_pdf_viewer(
    document_name: str,
    file_bytes: bytes,
    page_number: int,
    highlight_query: str = "",
) -> None:
    """Render one real PDF page and highlight matching query terms."""

    safe_name = html.escape(document_name)

    try:
        pdf_document = fitz.open(
            stream=file_bytes,
            filetype="pdf",
        )
    except Exception as error:
        st.error(f"Could not open the PDF viewer: {error}")
        return

    if pdf_document.page_count == 0:
        st.warning("This PDF has no pages.")
        pdf_document.close()
        return

    page_number = min(
        max(1, int(page_number)),
        pdf_document.page_count,
    )

    page = pdf_document.load_page(page_number - 1)
    highlighted_matches = 0

    for term in extract_query_terms(highlight_query):
        for rectangle in page.search_for(term):
            annotation = page.add_highlight_annot(rectangle)
            annotation.update()
            highlighted_matches += 1

    zoom = 1.65
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False,
    )
    image_bytes = pixmap.tobytes("png")
    pdf_document.close()

    st.markdown(
        f'<div class="viewer-title">📄 {safe_name} · Page {page_number}</div>',
        unsafe_allow_html=True,
    )

    if highlight_query:
        if highlighted_matches:
            st.caption(
                f"Highlighted {highlighted_matches} matching phrase(s) "
                "from the selected source question."
            )
        else:
            st.caption(
                "The cited page is shown. No exact query-word match "
                "was found to highlight on this page."
            )

    st.image(
        image_bytes,
        use_container_width=True,
    )


initialize_session_state()


with st.sidebar:
    st.title("📚 Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    uploaded_files = uploaded_files or []

    st.divider()

    column_one, column_two = st.columns(2)

    with column_one:
        if st.button(
            "➕ New chat",
            use_container_width=True,
        ):
            clear_conversation()
            st.rerun()

    with column_two:
        if st.button(
            "🗑️ Clear",
            use_container_width=True,
        ):
            clear_conversation()
            st.rerun()

    st.divider()
    st.caption("Local AI model")
    st.code("gemma3:4b", language=None)

    st.caption(
        "Documents and questions are processed locally through Ollama."
    )


st.title("📄 DocuMind AI")

st.caption(
    "A privacy-first document intelligence platform for text-based PDFs."
)


if not uploaded_files:
    st.markdown(
        """
<div class="hero-card">
    <h3>Upload documents to begin</h3>
    <p>
        Add one or more text-based PDFs from the sidebar.
        Your documents remain on your computer.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()


try:
    current_signature = create_file_signature(uploaded_files)

    if (
        st.session_state.active_document_signature
        and current_signature
        != st.session_state.active_document_signature
    ):
        clear_conversation()
        st.session_state.active_pdf_name = None
        st.session_state.active_pdf_page = 1

    st.session_state.active_document_signature = current_signature

    file_data = tuple(
        (
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
        for uploaded_file in uploaded_files
    )

    file_bytes_by_name = {
        filename: file_bytes
        for filename, file_bytes in file_data
    }

    if (
        st.session_state.active_pdf_name
        not in file_bytes_by_name
    ):
        st.session_state.active_pdf_name = uploaded_files[0].name
        st.session_state.active_pdf_page = 1
        st.session_state.active_highlight_query = ""

    with st.spinner(
        "Reading documents and building the search index..."
    ):
        (
            pages,
            chunks,
            document_details,
            vector_store,
        ) = build_document_index(file_data)

    if not chunks or vector_store is None:
        st.warning(
            "No readable text was found. The PDFs may be scanned, "
            "image-only, empty, or password protected."
        )
        st.stop()

    rag_engine = load_rag_engine()

    with st.sidebar:
        st.success(f"{len(uploaded_files)} document(s) ready")

        metric_one, metric_two = st.columns(2)

        with metric_one:
            st.metric("Pages", len(pages))

        with metric_two:
            st.metric("Chunks", len(chunks))

        st.divider()
        st.subheader("Uploaded files")

        for document in document_details:
            st.markdown(
                f"""
<div class="document-card">
    <div class="document-name">
        📄 {html.escape(document["name"])}
    </div>
    <div class="document-details">
        {document["pages"]} page(s) ·
        {document["chunks"]} chunks ·
        {document["size_kb"]:.1f} KB
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

        selected_pdf = st.selectbox(
            "PDF viewer",
            options=list(file_bytes_by_name),
            index=list(file_bytes_by_name).index(
                st.session_state.active_pdf_name
            ),
        )

        if selected_pdf != st.session_state.active_pdf_name:
            st.session_state.active_pdf_name = selected_pdf
            st.session_state.active_pdf_page = 1
            st.rerun()

        st.session_state.active_pdf_page = st.number_input(
            "Page",
            min_value=1,
            value=int(st.session_state.active_pdf_page),
            step=1,
        )

    chat_column, viewer_column = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with chat_column:
        st.markdown(
            """
<div class="hero-card">
    <h3>Ask your documents</h3>
    <p>
        Summarize, compare, extract facts, or ask follow-up questions.
        Every answer is grounded in retrieved document evidence.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        display_chat_history()

        if not st.session_state.messages:
            st.subheader("Suggested questions")

            suggested_questions = [
                "Summarize the main points in this document.",
                "What are the most important facts and numbers?",
                "What conclusions or recommendations are included?",
                "What dates, deadlines, or requirements are mentioned?",
                "Compare the main points across all uploaded documents.",
                "What information appears in more than one document?",
            ]

            suggestion_columns = st.columns(2)

            for index, suggestion in enumerate(suggested_questions):
                with suggestion_columns[index % 2]:
                    if st.button(
                        suggestion,
                        key=f"suggestion_{index}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = suggestion
                        st.rerun()

        typed_question = st.chat_input(
            "Ask a question about the uploaded documents"
        )

        question = (
            typed_question
            or st.session_state.pending_question
        )

        if question:
            st.session_state.pending_question = None
            conversation_history = get_conversation_history()

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user", avatar="👤"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🤖"):
                requested_top_k = get_top_k(
                    question=question,
                    total_chunks=len(chunks),
                )

                with st.status(
                    "Searching and reranking evidence...",
                    expanded=False,
                ) as status:
                    initial_results = vector_store.search(
                        query=question,
                        top_k=requested_top_k,
                    )

                    reranked_results = rag_engine.rerank_results(
                        question=question,
                        search_results=initial_results,
                        top_n=min(6, len(initial_results)),
                    )

                    status.update(
                        label="Generating answer locally...",
                        state="running",
                    )

                    token_stream = rag_engine.stream_answer(
                        question=question,
                        search_results=reranked_results,
                        conversation_history=conversation_history,
                    )

                    answer = st.write_stream(
                        stream_to_screen(token_stream)
                    )

                    status.update(
                        label="Answer complete",
                        state="complete",
                    )

                source_data = build_source_data(
                    reranked_results
                )

                # Automatically open the strongest cited source and highlight
                # both the user's question and the generated answer.
                if source_data:
                    strongest_source = source_data[0]
                    st.session_state.active_pdf_name = (
                        strongest_source["document"]
                    )
                    st.session_state.active_pdf_page = int(
                        strongest_source["page"]
                    )
                    st.session_state.active_highlight_query = (
                        f"{question} {answer}"
                    )

                display_sources(
                    source_data,
                    question,
                    key_prefix=f"current_{len(st.session_state.messages)}",
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_data,
                    "question": question,
                }
            )

    with viewer_column:
        render_pdf_viewer(
            document_name=st.session_state.active_pdf_name,
            file_bytes=file_bytes_by_name[
                st.session_state.active_pdf_name
            ],
            page_number=int(
                st.session_state.active_pdf_page
            ),
            highlight_query=st.session_state.active_highlight_query,
        )

except Exception as error:
    st.error("DocuMind encountered an error.")
    st.exception(error)