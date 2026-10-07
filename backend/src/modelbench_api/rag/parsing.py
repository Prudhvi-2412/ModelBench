from dataclasses import dataclass
from io import BytesIO


@dataclass(frozen=True)
class ParsedPage:
    text: str
    page_number: int | None


def parse_document(filename: str, data: bytes) -> list[ParsedPage]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in {"txt", "md"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Text documents must be UTF-8 encoded") from exc
        return [ParsedPage(text=text, page_number=None)]
    if suffix == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        return [
            ParsedPage(text=page.extract_text() or "", page_number=index)
            for index, page in enumerate(reader.pages, start=1)
        ]
    raise ValueError("Supported file types are .txt, .md, and .pdf")
