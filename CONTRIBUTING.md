# CONTRIBUTING.md

# Contributing to WhatUpDoc

## Branching Strategy

Never commit directly to `main`. All work happens on feature branches.

### Branch Naming

```
feature/your-feature-name
fix/bug-description
docs/what-youre-documenting
```

Examples:
```
feature/ingestion-pipeline
feature/chromadb-retriever
fix/chunking-overlap
docs/architecture
```

### Workflow

1. Pull the latest main before starting any work:
```bash
git checkout main
git pull origin main
```

2. Create your feature branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes with meaningful commit messages:
```bash
git add .
git commit -m "Add your meaningful commit message here"
```

4. Push your branch to GitHub:
```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request on GitHub into `main` and request a review from at least one teammate before merging.

---

## Commit Message Guidelines

Be specific. Grader and teammates should understand what changed without opening the diff.

**Good:**
```
Add ChromaDB persistent client and cosine similarity query
Implement fixed-size chunking with 64-token overlap
```

**Bad:**
```
updates
fix stuff
```

---

## Team Responsibilities

| Member | Branch(es) |
|--------|------------|
| Sean Costello | `feature/ingestion-pipeline` |
| Kat Fountain | `feature/evaluation-framework` |
| Claudia Porto | `feature/dataset-curation`, `feature/performance-testing` |
| Christopher Swartz | `feature/llm-infrastructure` |

---

## Pull Request Checklist

Before opening a PR, confirm:
- [ ] Code runs without errors
- [ ] Functions are documented with docstrings
- [ ] `requirements.txt` updated if new dependencies were added
- [ ] PR description explains what was changed and why
