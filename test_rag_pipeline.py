import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rag_pipeline import answer, chunks, ingest, retrieve, local_embed


class RagTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'docs'
        self.root.mkdir()
        self.db = Path(self.tmp.name) / 'index.sqlite3'
        (self.root / 'policy.md').write_text('Refunds are available for seven days after purchase. Contact support for refund requests.')
        (self.root / 'setup.txt').write_text('Install the package with pip. Run tests with python -m unittest.')

    def test_overlap(self):
        parts = chunks('alpha beta gamma delta epsilon zeta', 18, 5)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(parts))

    def test_bad_chunk_config(self):
        with self.assertRaises(ValueError): chunks('hi', 2, 2)

    def test_hash_embedding_deterministic(self):
        self.assertEqual(local_embed('refund days'), local_embed('refund days'))

    def test_ingest_retrieve_cite(self):
        result = ingest(self.root, self.db)
        self.assertEqual(result['files'], 2)
        hits = retrieve('refund requests', self.db)
        self.assertEqual(hits[0]['path'], 'policy.md')
        out = answer('refund requests', self.db)
        self.assertEqual(out['mode'], 'retrieval_only')
        self.assertEqual(out['citations'][0]['ref'], '[1]')

    def test_no_evidence(self):
        ingest(self.root, self.db)
        self.assertEqual(answer('zxqv hrrn', self.db)['mode'], 'no_evidence')

    def test_reingest_removes_deleted_files(self):
        ingest(self.root, self.db)
        (self.root / 'setup.txt').unlink()
        ingest(self.root, self.db)
        self.assertEqual({h['path'] for h in retrieve('package install', self.db)}, {'policy.md'})

    def test_symlink_skipped(self):
        (self.root / 'secret.md').symlink_to(self.root / 'policy.md')
        self.assertEqual(ingest(self.root, self.db)['files'], 2)

    def test_provider_mismatch(self):
        ingest(self.root, self.db)
        with self.assertRaises(ValueError): retrieve('refund', self.db, 'openai', 'key')

    def test_empty_index(self):
        self.assertEqual(retrieve('refund', self.db), [])

    def test_bad_query_bounds(self):
        with self.assertRaises(ValueError): retrieve(' ', self.db)
        with self.assertRaises(ValueError): retrieve('test', self.db, top_k=21)

    def test_openai_requires_key(self):
        with self.assertRaises(ValueError): ingest(self.root, self.db, 'openai')

    def test_generated_citation_validation(self):
        with patch('rag_pipeline.embed', side_effect=lambda texts, provider, key: [local_embed(t) for t in texts]):
            ingest(self.root, self.db, 'openai', 'key')
            with patch('rag_pipeline.openai_post', return_value={'choices':[{'message':{'content':'Seven days [9]'}}]}):
                result = answer('refund days', self.db, 'openai', 'key')
                self.assertEqual(result['mode'], 'citation_check_failed')
            with patch('rag_pipeline.openai_post', return_value={'choices':[{'message':{'content':'Seven days [1]'}}]}):
                result = answer('refund days', self.db, 'openai', 'key')
                self.assertEqual(result['mode'], 'generated')
