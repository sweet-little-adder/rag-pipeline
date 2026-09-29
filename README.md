# rag-pipeline

A small RAG pipeline you can run and inspect. It reads local Markdown and text files, makes overlapping chunks, embeds them, stores them in SQLite, retrieves nearest chunks, and returns source paths and chunk IDs. An optional OpenAI mode writes an answer with inline citations.

The default runs offline with no key or paid API. Its "embedding" is a hashed word-count vector, so it finds overlapping terms, not deep semantic matches. It returns excerpts, **not** a generated answer. Don't call that a production semantic search service.

## Run it

Python 3.10+, standard library only for the CLI and tests:

```sh
python3 -m unittest -v
mkdir -p /tmp/rag-pipeline-docs && cp sample.md /tmp/rag-pipeline-docs/
python3 rag_pipeline.py --db /tmp/rag-pipeline.sqlite3 ingest /tmp/rag-pipeline-docs --chunk-size 700 --overlap 120
python3 rag_pipeline.py --db /tmp/rag-pipeline.sqlite3 ask 'What happens if the lantern glass cracks?'
```

This repository has `sample.md` as a tiny fixture. Run `python3 evaluate.py` for a three-question retrieval@1 smoke test (synthetic, not a production benchmark). For your own data, point `ingest` at a directory of `.md` and `.txt` files. The ingest command recursively scans those formats, skips symlinks and atomically replaces the old index after embedding all chunks. It is a single-writer local workflow, not a web crawler. Each answer returns `mode`, `answer`, and `citations` (`path`, zero-based chunk number, score, excerpt).

For actual semantic embeddings and a generated answer, set `OPENAI_API_KEY` in your environment and run both commands with `--provider openai`. This calls `text-embedding-3-small` for indexing and retrieval and `gpt-4o-mini` for generation. API calls may cost money. No key goes into this repo. A provider mismatch requires re-ingest.

```sh
export OPENAI_API_KEY=... # set this privately; never commit a key
python3 rag_pipeline.py --db /tmp/rag-pipeline-openai.sqlite3 --provider openai ingest ./my-docs
python3 rag_pipeline.py --db /tmp/rag-pipeline-openai.sqlite3 --provider openai ask 'Your question'
```

## What is checked

`python3 -m unittest -v` covers chunking and invalid overlap, deterministic offline vectors, retrieval, citation structure, no-evidence behavior, re-index of deleted files, symlink exclusion, provider mismatch, empty index, input bounds, missing key, and a mocked generated-answer citation check. No paid model call is part of the tests. The OpenAI path is implemented but has not been smoke-tested with a live API key.

The generated answer is told to use only supplied excerpts and cite `[1]`, `[2]`, etc. A response with missing or out-of-range citations is rejected. This is syntax checking, **not** proof that every claim is supported. Retrieved document text is untrusted; prompt injection may still affect a model. Don't feed secrets or untrusted documents to a hosted model without your own data policy. Source paths are relative to the ingested directory. This tool isn't multi-tenant: anyone who can read the SQLite file can read its stored text.

## Next, if it mattered

Move retrieval to a vector index when the corpus outgrows in-memory scoring; add hybrid keyword search and an eval set with real questions and judged passages; check factual support claim by claim. This repo is deliberately narrow. It is not a claim of measured production quality.
