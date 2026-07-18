# Models

Model weights are **not** stored in this repository. All models are
managed by Ollama and stored locally in Ollama's own cache
(`~/.ollama/models`). Pull them once, offline forever after:

```bash
ollama pull llama3:8b          # primary generator (~4.7 GB, Q4)
ollama pull mistral:7b         # fallback generator (~4.1 GB, Q4)
ollama pull nomic-embed-text   # embedding model (~270 MB)
```

Selection rationale and benchmarking against alternatives:
see `docs/methods_research.md` §3.
