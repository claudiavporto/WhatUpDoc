# test_txt_ingestion.py
from utils.doc_ingestion import ingest_document

test_file = "data/corpus/medical/clinical_note_txt_001.txt"

for strategy in ["fixed", "sentence", "paragraph"]:
    print(f"\n=== Strategy: {strategy} ===")
    chunks, metadata = ingest_document(test_file, strategy=strategy)
    print(f"Metadata: {metadata}")
    print(f"Number of chunks: {len(chunks)}")
    if chunks:
        print(f"First chunk text (first 150 chars): {chunks[0]['text'][:150]}")
        print(f"First chunk metadata: {chunks[0]['metadata']}")