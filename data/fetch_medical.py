import hashlib
from datasets import load_dataset
from pathlib import Path
from docx import Document

DOCX_COUNT = 100
TXT_COUNT = 100
MIN_CHARS = 50
MAX_RECORDS_SCANNED = 3000

out = Path("data/corpus/medical")
out.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train", streaming=True)

seen_hashes = set()
docx_saved = 0
txt_saved = 0
scanned = 0
skipped_duplicate = 0
skipped_too_short = 0

for record in dataset:
    scanned += 1
    if scanned > MAX_RECORDS_SCANNED:
        print(f"Hit scan limit ({MAX_RECORDS_SCANNED}) before reaching targets.")
        break
    if docx_saved >= DOCX_COUNT and txt_saved >= TXT_COUNT:
        break

    text = record.get("full_note") or record.get("note")
    if text is None:
        print(f"  [WARN] record {scanned}: no 'full_note' or 'note' field, keys={list(record.keys())}")
        continue

    text = text.strip()

    if len(text) < MIN_CHARS:
        skipped_too_short += 1
        continue

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash in seen_hashes:
        skipped_duplicate += 1
        continue
    seen_hashes.add(text_hash)

    if docx_saved < DOCX_COUNT:
        docx_saved += 1
        doc = Document()
        doc.add_paragraph(text)
        doc.save(out / f"clinical_note_{docx_saved:03d}.docx")
        print(f"Saved .docx {docx_saved}/{DOCX_COUNT} (source record #{scanned})")
    elif txt_saved < TXT_COUNT:
        txt_saved += 1
        (out / f"clinical_note_txt_{txt_saved:03d}.txt").write_text(text, encoding="utf-8")
        print(f"Saved .txt {txt_saved}/{TXT_COUNT} (source record #{scanned})")

print()
print(f"Done. Saved: {docx_saved} .docx, {txt_saved} .txt, scanned: {scanned}, "
      f"skipped as duplicate: {skipped_duplicate}, skipped as too short: {skipped_too_short}")

if docx_saved < DOCX_COUNT or txt_saved < TXT_COUNT:
    print(f"WARNING: only got {docx_saved}/{DOCX_COUNT} .docx and {txt_saved}/{TXT_COUNT} .txt "
          f"before hitting the scan limit.")