from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import fitz


@dataclass(frozen=True)
class DocumentPage:
    """
    Plain Python representation of one extracted PDF page.
    """

    document_name: str
    page_number: int
    text: str


# Backward-compatible alias in case any older code imports PDFPage.
PDFPage = DocumentPage


def _read_pdf_bytes(
    uploaded_file: BinaryIO | BytesIO | bytes,
) -> bytes:
    """
    Read the complete PDF into memory without depending on the caller's
    file object remaining open.
    """

    if isinstance(uploaded_file, bytes):
        return uploaded_file

    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        original_position = None

        if hasattr(uploaded_file, "tell"):
            try:
                original_position = uploaded_file.tell()
            except Exception:
                original_position = None

        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

        data = uploaded_file.read()

        if (
            original_position is not None
            and hasattr(uploaded_file, "seek")
        ):
            try:
                uploaded_file.seek(original_position)
            except Exception:
                pass

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            "The uploaded PDF could not be read as bytes."
        )

    return bytes(data)


def extract_pdf_pages(
    uploaded_file: BinaryIO | BytesIO | bytes,
    document_name: str,
) -> list[DocumentPage]:
    """
    Extract text from every readable page of a PDF.

    The PyMuPDF document remains open until every page has been read.
    """

    pdf_bytes = _read_pdf_bytes(uploaded_file)

    if not pdf_bytes:
        return []

    document = None

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

        if document.is_encrypted:
            authenticated = document.authenticate("")

            if not authenticated:
                raise ValueError(
                    f"'{document_name}' is password protected. "
                    "Please upload an unlocked PDF."
                )

        extracted_pages: list[DocumentPage] = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = page.get_text("text").strip()

            extracted_pages.append(
                DocumentPage(
                    document_name=document_name,
                    page_number=page_index + 1,
                    text=text,
                )
            )

        return extracted_pages

    except fitz.FileDataError as error:
        raise ValueError(
            f"'{document_name}' is not a valid readable PDF."
        ) from error

    except RuntimeError as error:
        raise ValueError(
            f"PyMuPDF could not read '{document_name}'. "
            "The file may be damaged or password protected."
        ) from error

    finally:
        if document is not None:
            document.close()