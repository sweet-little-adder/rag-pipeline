"""Small retrieval recall smoke test, not answer quality measurement."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copyfile
from rag_pipeline import ingest, retrieve

source = Path(__file__).parent
cases = json.loads((source / 'eval.json').read_text())
with TemporaryDirectory() as temp:
    root = Path(temp) / 'docs'
    root.mkdir()
    copyfile(source / 'sample.md', root / 'sample.md')
    db = Path(temp) / 'index.sqlite3'
    ingest(root, db)
    score = 0
    for case in cases:
        top = retrieve(case['question'], db, top_k=1)[0]
        passed = top['path'] == case['expected_path'] and all(
            phrase in top['text'].lower() for phrase in case['expected_terms'])
        score += passed
        print(f"{'PASS' if passed else 'FAIL'} {case['question']} -> {top['path']}#chunk-{top['chunk']}")
    print(f'retrieval@1: {score}/{len(cases)} (small synthetic fixture; not production accuracy)')
    if score != len(cases):
        raise SystemExit(1)
