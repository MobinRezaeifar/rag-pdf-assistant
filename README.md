# RAG PDF Assistant 🚀

A tiny, ultra-fast Retrieval-Augmented Generation (RAG) assistant for PDFs.  
Ask questions in the terminal and get instant answers from your documents.

## Highlights
- ⚡ Index once, query instantly (FAISS)
- 🧠 High-quality embeddings (`bge-small-en` via Sentence Transformers)
- 🔗 Load PDFs from **local folder or any direct URL**
- 🧾 **URL downloads keep the original filename** (e.g., `?fileName=foo.pdf`), with smart de-duplication (`foo_1.pdf`, `foo_2.pdf`, ...)
- 🗂️ Per-file vector DB under `db/<pdf-stem>` for clean separation
- ⌨️ **Polished CLI**: choose Local / URL / Exit; **type `exit`** (or press Enter on an empty line) to leave Q&A
- 🧩 Minimal codebase, easy to extend

---

## Quickstart

```bash
git clone https://github.com/MobinRezaeifar/rag-pdf-assistant.git
cd rag-pdf-assistant
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Put your PDF
Place your PDF files in the `./data/` folder, for example:
```
data/
  nist.ai.100-1.pdf
```
You can also enter a direct URL when prompted; the file will be saved into `./data/` using the original filename when possible.

---

## Usage

### 1) Simple run (auto index + interactive Q&A)
When you run the script, it asks whether you want to:
- Choose a **local PDF file** from the `./data` folder
- **Enter a direct PDF link** to download and analyze
- **Exit**

Downloaded PDFs are saved automatically and indexed for future use.

```bash
python main.py
```

### Example session
```
Select PDF source:
  1) Local file in ./data
  2) Link (URL)
  3) Exit
Enter 1, 2 or 3 [default 1]: 2
Enter PDF URL: https://example.com/download?fileName=guideline.pdf
⬇️  Downloading PDF ...
✅ Saved to data/guideline.pdf
✅ Loaded 139 pages from data/guideline.pdf.
🔧 Building vector index from PDF ...
✅ Indexed 139 chunks.

💬 RAG ready. Ask your question about the PDF.
   (Type 'exit' or press Enter on empty line to exit.)

❓ Question: Copyright
🔎 Top matches:
1. [Page 2] ...

❓ Question: exit
👋 Bye.
```

---

## Project Structure
```
rag-pdf-assistant/
  data/               # Put your PDFs here
  db/                 # Auto-created FAISS index + metas (per-PDF subfolder)
  utils/
    embedder.py       # Vector store (FAISS) + encoding
    pdf_loader.py     # PDF to text (page-level)
  main.py             # CLI logic (with Local/URL/Exit + interactive QA)
  requirements.txt
  .gitignore
  README.md
```

---

## How it works
1) Load PDF pages (PyPDF) → list of strings  
2) Encode with Sentence Transformers (bge-small-en) → embeddings  
3) Normalize + index in FAISS (cosine/IP)  
4) Query: embed question → FAISS top-k → print best matches with page info

---

## Configuration
- `MODEL_NAME` (in `main.py`): embedding model path or name (default: `models/bge-small-en`)
- Vector DB location: `db/<pdf-stem>` (auto-created)
- Retrieval count: `top_k=5` in `interactive_qa`

> Tip: If you change the embedding model (e.g., to `BAAI/bge-small-en-v1.5`), the index will be rebuilt the next time you run.

---

## Common operations

### Rebuild an index for a PDF
Delete the corresponding folder under `db/<pdf-stem>` and run again; the index will be re-created automatically.

### Exit the program
- In the **menu**: choose `3) Exit`.
- In the **Q&A loop**: type `exit` or press **Enter** on an empty line. You can also use `Ctrl+C`.

---

## Requirements
- Python 3.10+
- Packages in `requirements.txt`:
  - langchain
  - chromadb
  - transformers
  - sentence-transformers
  - faiss-cpu
  - pdfminer.six
  - tqdm
  - pypdf
  - numpy
  - requests

Install with:
```bash
pip install -r requirements.txt
```

---

## Tips
- `db/` is auto-created and should not be committed to git.
- For large PDFs, consider chunking by section (current default is per-page). This repo aims to be tiny; feel free to extend!

---

## License
MIT
