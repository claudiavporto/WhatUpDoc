# quick_debug_empty_chunks.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.doc_ingestion import ingest_document

DOMAINS_TO_CHECK = ["data/corpus/legal", "data/corpus/policy"]
EXTENSIONS = {".pdf", ".docx"}

found_any = False

for domain_path in DOMAINS_TO_CHECK:
    domain_dir = Path(domain_path)
    files = [f for f in sorted(domain_dir.rglob("*")) if f.suffix.lower() in EXTENSIONS]

    for file_path in files:
        try:
            chunks, metadata = ingest_document(str(file_path), strategy="sentence")
        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")
            continue

        for i, c in enumerate(chunks):
            word_count = len(c["text"].split())
            if word_count <= 2:  # catch near-empty too, not just exactly 0
                found_any = True
                print(f"\nFile: {file_path}")
                print(f"  Chunk {i}: word_count={word_count}")
                print(f"  repr(text) = {repr(c['text'])}")
                print(f"  metadata = {c['metadata']}")

if not found_any:
    print("\nNo empty/near-empty chunks found in legal or policy under 'sentence' strategy.")