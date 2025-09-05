from utils.pdf_loader import load_pdf
from utils.embedder import Embedder
import os
import sys
from pathlib import Path
import requests
from urllib.parse import urlparse, parse_qs, unquote

MODEL_NAME = "models/bge-small-en"
DEFAULT_LOCAL_DIR = "data"

def ensure_index(embedder, pages):
    need_build = (
        not os.path.exists(embedder.index_file)
        or not os.path.exists(embedder.meta_file)
        or getattr(embedder.index, "ntotal", 0) == 0
    )
    if need_build:
        print("🔧 Building vector index from PDF ...")
        embedder.encode_and_store(pages)
        print(f"✅ Indexed {getattr(embedder.index, 'ntotal', 0)} chunks.")
    else:
        print(f"✅ Loaded index with {embedder.index.ntotal} vectors.")

def interactive_qa(embedder, top_k=5):
    print("\n💬 RAG ready. Ask your question about the PDF.")
    print("   (Type 'exit' or press Enter on empty line to exit.)")
    while True:
        try:
            q = input("\n❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye.")
            break
        if not q or q.lower() == "exit":
            print("👋 Bye.")
            break
        hits = embedder.search(q, top_k=top_k)
        if not hits:
            print("— No results. Try another query.")
            continue
        print("\n🔎 Top matches:")
        for i, h in enumerate(hits, 1):
            preview = h.replace("\n", " ").strip()
            if len(preview) > 400:
                preview = preview[:400] + " ..."
            print(f"{i}. {preview}")

def list_local_pdfs(directory):
    p = Path(directory)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    files = sorted([f.name for f in p.glob("*.pdf")])
    return files

def choose_local_pdf():
    files = list_local_pdfs(DEFAULT_LOCAL_DIR)
    if files:
        print("📄 Available PDFs in ./data:")
        for idx, name in enumerate(files, 1):
            print(f"  {idx}. {name}")
        choice = input(f"Enter number or name [1-{len(files)}] (default 1): ").strip()
        if not choice:
            filename = files[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(files):
            filename = files[int(choice) - 1]
        else:
            filename = choice
        pdf_path = Path(DEFAULT_LOCAL_DIR) / filename
    else:
        filename = input("No PDFs found in ./data. Enter a filename to use (will look under ./data): ").strip()
        pdf_path = Path(DEFAULT_LOCAL_DIR) / filename
    return str(pdf_path)

def safe_filename_from_url(url):
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    name = None
    if "fileName" in qs and qs["fileName"]:
        name = unquote(qs["fileName"][0])
    if not name:
        name = os.path.basename(parsed.path)
    if not name:
        name = "downloaded.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name

def download_pdf(url, dest_dir="data"):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    name = safe_filename_from_url(url)
    target = Path(dest_dir) / name
    base = target.stem
    ext = target.suffix
    i = 1
    while target.exists():
        target = Path(dest_dir) / f"{base}_{i}{ext}"
        i += 1
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(target, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return str(target)

def select_pdf_source():
    print("Select PDF source:")
    print("  1) Local file in ./data")
    print("  2) Link (URL)")
    print("  3) Exit")
    choice = input("Enter 1, 2 or 3 [default 1]: ").strip() or "1"
    if choice == "3":
        print("👋 Bye.")
        sys.exit(0)
    if choice == "2":
        while True:
            url = input("Enter PDF URL: ").strip()
            if url:
                print("⬇️  Downloading PDF ...")
                path = download_pdf(url, DEFAULT_LOCAL_DIR)
                print(f"✅ Saved to {path}")
                return path
            else:
                print("URL is required.")
    else:
        path = choose_local_pdf()
        return path

if __name__ == "__main__":
    try:
        pdf_path = select_pdf_source()
        if not Path(pdf_path).exists():
            print(f"❌ PDF not found: {pdf_path}")
            sys.exit(1)
        pages = load_pdf(pdf_path)
        print(f"✅ Loaded {len(pages)} pages from {pdf_path}.")
        db_subdir = Path("db") / Path(pdf_path).stem
        emb = Embedder(model_name=MODEL_NAME, db_path=str(db_subdir))
        emb.load()
        ensure_index(emb, pages)
        interactive_qa(emb, top_k=5)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
