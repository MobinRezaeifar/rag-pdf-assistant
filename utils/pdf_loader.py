from pypdf import PdfReader

def load_pdf(file_path: str) -> list[str]:
    reader = PdfReader(file_path)
    texts = []
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text()
        if t:
            t = t.strip()
        if t:
            texts.append(f"[Page {i}] {t}")
    return texts

