#!/usr/bin/env python3
"""
build_query_candidates.py

Scans the WhatUpDoc corpus (policy, legal, medical) and auto-generates
CANDIDATE test queries with known-correct answers, for RQ1 (retrieval
precision) evaluation across all three categories.

This does NOT produce a final query set. It produces a CSV of candidates,
each tied to a specific source file, category, and location (page for
PDFs, paragraph batch for DOCX/TXT), for a human (you, Kat, the team) to
review, correct, and trim down to the final query set.

File types by category:
  policy   -> PDF          (EPA drinking water regulations)
  legal    -> PDF          (CUAD contracts)
  medical  -> DOCX and TXT (clinical notes)

Per-category patterns:
  policy:   numeric standards, defined terms, compliance dates, section headers
  legal:    defined terms, dollar amounts, notice/termination periods, governing law
  medical:  diagnoses, medications + dosage, visit dates, lab values

Requires: pymupdf, python-docx
    pip install pymupdf python-docx --break-system-packages

Usage:
    python build_query_candidates.py --data-dir data
    python build_query_candidates.py --data-dir data --max-per-file 6
    python build_query_candidates.py --data-dir data --only policy legal
"""

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF is required. Install with:")
    print("  pip install pymupdf --break-system-packages")
    sys.exit(1)

try:
    import docx
except ImportError:
    print("ERROR: python-docx is required. Install with:")
    print("  pip install python-docx --break-system-packages")
    sys.exit(1)


CATEGORY_FOLDER_HINTS = {
    "policy": ["policy"],
    "legal": ["legal"],
    "medical": ["medical"],
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------------
# POLICY patterns (PDF)
# ----------------------------------------------------------------------

POLICY_NUMERIC_RE = re.compile(
    r"(?P<subject>[A-Z][A-Za-z0-9 ,'\-]{3,80}?)\s+"
    r"(?:of|is|shall be|shall not exceed|not to exceed)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg/L|\u00b5g/L|ug/L|mg/l|MFL|pCi/L|NTU|percent|%)",
    re.IGNORECASE,
)

POLICY_DEFINED_TERM_RE = re.compile(
    r"[\u201c\"']?(?P<term>[A-Z][A-Za-z0-9 \-]{2,60})[\u201d\"']?\s+means\s+"
    r"(?P<definition>[^.]{10,220}\.)",
)

POLICY_COMPLIANCE_DATE_RE = re.compile(
    r"(?P<context>[A-Za-z0-9 ,'\-]{10,100}?)\s+"
    r"(?:no later than|by|beginning|effective)\s+"
    r"(?P<date>(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},\s+\d{4})",
)

POLICY_SECTION_HEADER_RE = re.compile(
    r"^(?:Sec\.|\u00a7)\s*(?P<number>\d{2,3}\.\d{1,3})\s+(?P<title>[A-Z][A-Za-z0-9 ,\-/]{3,80})$",
    re.MULTILINE,
)


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation so regexes don't bleed subject
    text across unrelated sentences after newlines are flattened to spaces."""
    flat = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


# Generic openers like "This means..." or "The text means..." are ordinary
# sentences, not term definitions, but they match the same "X means Y"
# shape as a real defined term. Reject any captured "term" whose first word
# is one of these, rather than trusting the regex alone.
GENERIC_TERM_OPENERS = {
    "this", "that", "these", "those", "it", "the text", "the following",
    "such", "here",
}


def is_generic_term(term: str) -> bool:
    first_word = term.strip().lower().split(" ")[0]
    return first_word in GENERIC_TERM_OPENERS or term.strip().lower() in GENERIC_TERM_OPENERS


# A captured "term" ending in a function word (article, preposition, "to be"
# verb) means the regex grabbed a clause fragment, not an actual term, e.g.
# "OATT as the" or "DCF analysis as an appropriate".
TERM_TRAILING_STOPWORDS = {"a", "an", "the", "as", "is", "of", "to", "that", "this", "best"}


def ends_in_stopword(term: str) -> bool:
    last_word = term.strip().lower().split(" ")[-1]
    return last_word in TERM_TRAILING_STOPWORDS


# "X means to do Y" or "X means of doing Y" uses "means" as a noun (a method
# or way), not as the verb of a definition ("X means Y"). Real definitions
# are followed by a noun phrase, not an infinitive or a gerund-of-phrase.
IDIOMATIC_MEANS_RE = re.compile(r"^(?:to\s+\w+|of\s+\w+ing\b)", re.IGNORECASE)


def is_idiomatic_means(definition: str) -> bool:
    return bool(IDIOMATIC_MEANS_RE.match(definition.strip()))


def extract_policy(text: str, source_file: str, location: int) -> list[dict]:
    candidates = []
    flat = text.replace("\n", " ")

    for sentence in split_sentences(text):
        for m in POLICY_NUMERIC_RE.finditer(sentence):
            candidates.append({
                "query": f"What is the standard or limit for {clean(m.group('subject')).lower()}?",
                "expected_answer": f"{m.group('value')} {m.group('unit')}",
                "answer_context": clean(m.group(0)),
                "source_file": source_file, "category": "policy", "location": location,
                "match_type": "numeric_standard",
            })

    for m in POLICY_DEFINED_TERM_RE.finditer(flat):
        term = clean(m.group("term"))
        definition = clean(m.group("definition"))
        if len(term.split()) > 6 or is_generic_term(term) or ends_in_stopword(term) or is_idiomatic_means(definition):
            continue
        candidates.append({
            "query": f"What does '{term}' mean under this rule?",
            "expected_answer": clean(m.group("definition")),
            "answer_context": clean(m.group(0)),
            "source_file": source_file, "category": "policy", "location": location,
            "match_type": "defined_term",
        })

    for m in POLICY_COMPLIANCE_DATE_RE.finditer(flat):
        candidates.append({
            "query": f"By when must {clean(m.group('context')).lower()} comply?",
            "expected_answer": m.group("date"),
            "answer_context": clean(m.group(0)),
            "source_file": source_file, "category": "policy", "location": location,
            "match_type": "compliance_date",
        })

    for m in POLICY_SECTION_HEADER_RE.finditer(text):
        candidates.append({
            "query": f"What section covers {clean(m.group('title')).lower()}?",
            "expected_answer": f"Section {m.group('number')}: {clean(m.group('title'))}",
            "answer_context": clean(m.group(0)),
            "source_file": source_file, "category": "policy", "location": location,
            "match_type": "section_header",
        })

    return candidates


# ----------------------------------------------------------------------
# LEGAL patterns (PDF, CUAD contracts)
# ----------------------------------------------------------------------

LEGAL_DEFINED_TERM_RE = re.compile(
    r'"(?P<term>[A-Z][A-Za-z0-9 \-]{2,50})"\s+(?:shall mean|means)\s+'
    r"(?P<definition>[^.]{10,220}\.)",
)

LEGAL_DOLLAR_RE = re.compile(
    r"(?P<context>[A-Za-z0-9 ,'\-]{5,80}?)\s+of\s+"
    r"(?P<amount>\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|thousand))?)",
    re.IGNORECASE,
)

LEGAL_NOTICE_PERIOD_RE = re.compile(
    r"(?P<context>written notice|notice of termination|prior notice)\s+"
    r"of\s+(?:at least\s+)?(?P<days>\d{1,3})\s+days",
    re.IGNORECASE,
)

LEGAL_GOVERNING_LAW_RE = re.compile(
    r"governed by\s+(?:and construed in accordance with\s+)?the laws of\s+"
    r"(?:the State of\s+)?(?P<jurisdiction>[A-Z][A-Za-z ]{2,30})",
)

LEGAL_EFFECTIVE_DATE_RE = re.compile(
    r"(?:this agreement|effective as of|commencing)\s+.{0,40}?"
    r"(?P<date>(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE,
)


def extract_legal(text: str, source_file: str, location: int) -> list[dict]:
    candidates = []
    flat = text.replace("\n", " ")

    for m in LEGAL_DEFINED_TERM_RE.finditer(flat):
        term = clean(m.group("term"))
        if is_generic_term(term) or ends_in_stopword(term):
            continue
        candidates.append({
            "query": f"How is '{term}' defined in this agreement?",
            "expected_answer": clean(m.group("definition")),
            "answer_context": clean(m.group(0)),
            "source_file": source_file, "category": "legal", "location": location,
            "match_type": "defined_term",
        })

    for sentence in split_sentences(text):
        for m in LEGAL_DOLLAR_RE.finditer(sentence):
            candidates.append({
                "query": f"What is the {clean(m.group('context')).lower()}?",
                "expected_answer": m.group("amount"),
                "answer_context": clean(m.group(0)),
                "source_file": source_file, "category": "legal", "location": location,
                "match_type": "dollar_amount",
            })

    for m in LEGAL_NOTICE_PERIOD_RE.finditer(flat):
        candidates.append({
            "query": "How many days of notice does this agreement require for termination?",
            "expected_answer": f"{m.group('days')} days",
            "answer_context": clean(m.group(0)),
            "source_file": source_file, "category": "legal", "location": location,
            "match_type": "notice_period",
        })

    for m in LEGAL_GOVERNING_LAW_RE.finditer(flat):
        candidates.append({
            "query": "Which jurisdiction's laws govern this agreement?",
            "expected_answer": clean(m.group("jurisdiction")),
            "answer_context": clean(m.group(0)),
            "source_file": source_file, "category": "legal", "location": location,
            "match_type": "governing_law",
        })

    for m in LEGAL_EFFECTIVE_DATE_RE.finditer(flat):
        candidates.append({
            "query": "What is the effective date of this agreement?",
            "expected_answer": m.group("date"),
            "answer_context": clean(m.group(0)),
            "source_file": source_file, "category": "legal", "location": location,
            "match_type": "effective_date",
        })

    return candidates


# ----------------------------------------------------------------------
# MEDICAL patterns (DOCX, clinical notes)
# ----------------------------------------------------------------------

MEDICAL_DIAGNOSIS_RE = re.compile(
    r"(?:diagnosed with|diagnosis of|diagnosis:)\s+"
    r"(?P<diagnosis>[A-Za-z][A-Za-z0-9 ,\-]{3,150}?)(?=[.,;]|$)",
    re.IGNORECASE,
)

MEDICAL_MEDICATION_RE = re.compile(
    r"(?P<med>[A-Z][a-z]{2,25})\s+(?P<dose>\d{1,4}\s?(?:mg|mcg|mL|g|units))\b",
)

MEDICAL_VISIT_DATE_RE = re.compile(
    r"(?:date of (?:visit|admission)|admitted on|seen on)[:\s]+"
    r"(?P<date>(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)

MEDICAL_LAB_VALUE_RE = re.compile(
    r"(?P<test>[A-Za-z][A-Za-z0-9 \-]{2,100}?)\s+(?:level|value|result)?\s*(?:of|was|:)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg/dL|mmol/L|%|bpm|mmHg)",
    re.IGNORECASE,
)


def extract_medical(text: str, source_file: str, location: int) -> list[dict]:
    candidates = []
    sentences = split_sentences(text)

    for sentence in sentences:
        for m in MEDICAL_DIAGNOSIS_RE.finditer(sentence):
            candidates.append({
                "query": "What diagnosis is documented in this note?",
                "expected_answer": clean(m.group("diagnosis")),
                "answer_context": clean(m.group(0)),
                "source_file": source_file, "category": "medical", "location": location,
                "match_type": "diagnosis",
            })

        for m in MEDICAL_MEDICATION_RE.finditer(sentence):
            candidates.append({
                "query": f"What dose of {clean(m.group('med'))} was prescribed?",
                "expected_answer": clean(m.group("dose")),
                "answer_context": clean(m.group(0)),
                "source_file": source_file, "category": "medical", "location": location,
                "match_type": "medication_dose",
            })

        for m in MEDICAL_VISIT_DATE_RE.finditer(sentence):
            candidates.append({
                "query": "What was the date of this visit or admission?",
                "expected_answer": clean(m.group("date")),
                "answer_context": clean(m.group(0)),
                "source_file": source_file, "category": "medical", "location": location,
                "match_type": "visit_date",
            })

        for m in MEDICAL_LAB_VALUE_RE.finditer(sentence):
            candidates.append({
                "query": f"What was the {clean(m.group('test')).lower()} result?",
                "expected_answer": f"{m.group('value')} {m.group('unit')}",
                "answer_context": clean(m.group(0)),
                "source_file": source_file, "category": "medical", "location": location,
                "match_type": "lab_value",
            })

    return candidates


# ----------------------------------------------------------------------
# File processing
# ----------------------------------------------------------------------

EXTRACTORS = {
    "policy": extract_policy,
    "legal": extract_legal,
    "medical": extract_medical,
}


def detect_category(path: Path, data_dir: Path) -> str | None:
    parts = [p.lower() for p in path.relative_to(data_dir).parts]
    for category, hints in CATEGORY_FOLDER_HINTS.items():
        if any(hint in parts for hint in hints):
            return category
    return None


def process_pdf(path: Path, category: str, max_per_file: int) -> list[dict]:
    file_candidates = []
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  [SKIP] {path.name}: could not open ({e})")
        return []

    extractor = EXTRACTORS[category]
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue
        file_candidates.extend(extractor(text, path.name, page_num))
        if len(file_candidates) >= max_per_file:
            break

    doc.close()
    return file_candidates[:max_per_file]


def process_docx(path: Path, category: str, max_per_file: int) -> list[dict]:
    file_candidates = []
    try:
        d = docx.Document(path)
    except Exception as e:
        print(f"  [SKIP] {path.name}: could not open ({e})")
        return []

    extractor = EXTRACTORS[category]
    # Batch paragraphs into groups to give the regexes enough surrounding
    # context, since clinical notes are short per-paragraph.
    paras = [p.text for p in d.paragraphs if p.text.strip()]
    batch_size = 15
    for i in range(0, len(paras), batch_size):
        batch_text = "\n".join(paras[i:i + batch_size])
        location = (i // batch_size) + 1  # batch number, not a true page
        file_candidates.extend(extractor(batch_text, path.name, location))
        if len(file_candidates) >= max_per_file:
            break

    return file_candidates[:max_per_file]


def process_txt(path: Path, category: str, max_per_file: int) -> list[dict]:
    """Mirrors doc_ingestion.py's parse_txt: paragraphs are separated by a
    SINGLE newline (not a blank line), literal two-character "\\n" sequences
    are normalized to real newlines first, and decoding falls back to
    latin-1 if the file isn't valid UTF-8."""
    file_candidates = []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = path.read_text(encoding="latin-1")
    except Exception as e:
        print(f"  [SKIP] {path.name}: could not open ({e})")
        return []

    raw_text = raw_text.replace("\\n", "\n")
    paras = [p.strip() for p in raw_text.split("\n") if p.strip()]

    extractor = EXTRACTORS[category]
    batch_size = 15
    for i in range(0, len(paras), batch_size):
        batch_text = "\n".join(paras[i:i + batch_size])
        location = (i // batch_size) + 1  # batch number, not a true page
        file_candidates.extend(extractor(batch_text, path.name, location))
        if len(file_candidates) >= max_per_file:
            break

    return file_candidates[:max_per_file]


def dedupe(candidates: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for c in candidates:
        key = (c["category"], c["match_type"], c["expected_answer"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Root data folder containing policy/legal/medical subfolders")
    parser.add_argument("--out", default="query_candidates.csv", help="Output CSV path")
    parser.add_argument("--max-per-file", type=int, default=8, help="Cap candidates per source file")
    parser.add_argument("--only", nargs="+", choices=["policy", "legal", "medical"],
                         help="Restrict to specific categories (default: all three)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        sys.exit(1)

    active_categories = set(args.only) if args.only else set(EXTRACTORS.keys())

    pdfs = list(data_dir.rglob("*.pdf"))
    docxs = list(data_dir.rglob("*.docx"))
    txts = list(data_dir.rglob("*.txt"))

    print("=" * 70)
    print("QUERY CANDIDATE GENERATION (corpus-wide)")
    print("=" * 70)

    all_candidates = []
    counts_by_category = {c: 0 for c in EXTRACTORS}
    files_by_category = {c: 0 for c in EXTRACTORS}
    unmatched_files = []

    for path in sorted(pdfs):
        category = detect_category(path, data_dir)
        if category is None or category not in active_categories:
            if category is None:
                unmatched_files.append(path)
            continue
        file_candidates = process_pdf(path, category, args.max_per_file)
        print(f"  [{category:8s}] [{len(file_candidates):2d} candidates]  {path.name}")
        all_candidates.extend(file_candidates)
        counts_by_category[category] += len(file_candidates)
        files_by_category[category] += 1

    for path in sorted(docxs):
        category = detect_category(path, data_dir)
        if category is None or category not in active_categories:
            if category is None:
                unmatched_files.append(path)
            continue
        file_candidates = process_docx(path, category, args.max_per_file)
        print(f"  [{category:8s}] [{len(file_candidates):2d} candidates]  {path.name}")
        all_candidates.extend(file_candidates)
        counts_by_category[category] += len(file_candidates)
        files_by_category[category] += 1

    for path in sorted(txts):
        category = detect_category(path, data_dir)
        if category is None or category not in active_categories:
            if category is None:
                unmatched_files.append(path)
            continue
        file_candidates = process_txt(path, category, args.max_per_file)
        print(f"  [{category:8s}] [{len(file_candidates):2d} candidates]  {path.name}")
        all_candidates.extend(file_candidates)
        counts_by_category[category] += len(file_candidates)
        files_by_category[category] += 1

    before = len(all_candidates)
    all_candidates = dedupe(all_candidates)
    after = len(all_candidates)

    fieldnames = ["category", "query", "expected_answer", "answer_context",
                  "source_file", "location", "match_type", "reviewed", "reviewer_notes"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in all_candidates:
            row = dict(c)
            row["reviewed"] = ""
            row["reviewer_notes"] = ""
            writer.writerow(row)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PDFs found:                {len(pdfs)}")
    print(f"DOCX found:                {len(docxs)}")
    print(f"TXT found:                 {len(txts)}")
    if unmatched_files:
        print(f"Files with no category match (skipped): {len(unmatched_files)}")
        print("  (folder name did not contain 'policy', 'legal', or 'medical')")
    print(f"Raw candidates found:      {before}")
    print(f"After deduplication:       {after}")
    print()
    for category in ["policy", "legal", "medical"]:
        print(f"  {category:10s} files: {files_by_category[category]:3d}   candidates: {counts_by_category[category]}")
    print()
    print(f"Output written to:         {args.out}")
    print()
    print("NEXT STEP: open the CSV, review each row against the source document,")
    print("fix wording, mark 'reviewed' = yes/no, and trim to your final query")
    print("set. Aim for a roughly even split across the three categories so")
    print("RQ1 results generalize across the whole corpus, not just policy.")
    print("Discard anything the regex mismatched.")
    print("Done.")


if __name__ == "__main__":
    main()
