from datasets import load_dataset
from pathlib import Path
from docx import Document

dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train", streaming=True)

out = Path("data/sample/medical")
out.mkdir(parents=True, exist_ok=True)

for i, record in enumerate(dataset):
    if i >= 100:
        break
    text = record.get("full_note", record.get("note", str(record)))
    doc = Document()
    doc.add_paragraph(text)
    doc.save(out / f"clinical_note_{i+1:03d}.docx")
    print(f"Saved record {i+1}")

print("Done")