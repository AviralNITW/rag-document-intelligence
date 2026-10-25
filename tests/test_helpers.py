"""Unit tests for the pure logic helpers (no network, no model loading).

Run from the project root:
    python -m unittest discover -s tests -v
"""
import os
import re
import shutil
import tempfile
import unittest

from langchain_core.documents import Document

from services.conversation_service import create_new_conversation, generate_title_from_query
from services.document_service import (
    format_file_size,
    relative_time,
    filter_documents_by_format,
    collection_name_for,
    extract_document_pages,
    FORMAT_FILTER_MAP,
)
from services.rag_service import (
    bm25_search,
    reciprocal_rank_fusion,
    FALLBACK_MODELS,
    rerank_documents,
    should_rewrite_query,
    _bm25_for,
)
from services.workflow_service import select_representative_chunks
import services.session_store as session_store


def _doc(content: str, source: str = "a.pdf", page: int = 0) -> Document:
    return Document(page_content=content, metadata={"source": source, "page": page})


class TestFormatting(unittest.TestCase):
    def test_format_file_size(self):
        self.assertEqual(format_file_size(512), "512 B")
        self.assertEqual(format_file_size(2048), "2.0 KB")
        self.assertEqual(format_file_size(5 * 1024 * 1024), "5.0 MB")

    def test_relative_time(self):
        from datetime import datetime, timedelta

        now = datetime.now()
        self.assertEqual(relative_time(now.strftime("%Y-%m-%d %H:%M:%S")), "Just now")
        self.assertEqual(relative_time((now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")), "5m ago")
        self.assertEqual(relative_time((now - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")), "3h ago")
        self.assertEqual(relative_time((now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")), "2d ago")
        self.assertEqual(relative_time("not-a-date"), "—")
        self.assertEqual(relative_time(None), "—")


class TestFormatFilter(unittest.TestCase):
    DOCS = [
        {"name": "a.pdf", "type": "PDF"},
        {"name": "b.docx", "type": "DOCX"},
        {"name": "c.txt", "type": "TXT"},
        {"name": "d.md", "type": "MD"},
    ]

    def test_markdown_label_maps_to_md_type(self):
        # Regression: "Markdown" used to compare against "MARKDOWN" and always return [].
        result = filter_documents_by_format(self.DOCS, "Markdown")
        self.assertEqual([d["name"] for d in result], ["d.md"])

    def test_all_formats_returns_everything(self):
        self.assertEqual(len(filter_documents_by_format(self.DOCS, "All Formats")), 4)

    def test_pdf_filter(self):
        self.assertEqual([d["name"] for d in filter_documents_by_format(self.DOCS, "PDF")], ["a.pdf"])

    def test_unknown_label_is_safe(self):
        self.assertEqual(filter_documents_by_format(self.DOCS, "Nope"), [])

    def test_every_label_except_all_is_typed(self):
        for label, types in FORMAT_FILTER_MAP.items():
            if types is None:
                self.assertEqual(label, "All Formats")


class TestCollectionNaming(unittest.TestCase):
    def test_uuid_is_sanitized(self):
        name = collection_name_for("0246f01f-b55f-40a0-a49e-c4c6fb3f1f00")
        self.assertEqual(name, "session_0246f01f_b55f_40a0_a49e_c4c6fb3f1f00")
        self.assertNotIn("-", name)


class TestBM25Scoping(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            _doc("python programming language features", source="py.pdf"),
            _doc("cooking pasta with tomato sauce", source="food.pdf"),
            _doc("python decorators and generators", source="py2.pdf"),
        ]

    def test_scope_restricts_results(self):
        results = bm25_search("python", self.chunks, selected_docs=["py.pdf"], top_k=10)
        self.assertTrue(results)
        self.assertTrue(all(r.metadata["source"] == "py.pdf" for r in results))

    def test_empty_scope_never_widens_to_all_docs(self):
        # Regression: an unmatched scope used to silently fall back to every chunk.
        results = bm25_search("python", self.chunks, selected_docs=["missing.pdf"], top_k=10)
        self.assertEqual(results, [])

    def test_no_scope_returns_matches(self):
        results = bm25_search("python", self.chunks, top_k=10)
        self.assertEqual(len(results), 2)

    def test_empty_chunks(self):
        self.assertEqual(bm25_search("python", []), [])

    def test_top_k_respected(self):
        results = bm25_search("python", self.chunks, top_k=1)
        self.assertEqual(len(results), 1)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_doc_in_both_lists_wins(self):
        dense = [_doc("dense doc", page=0), _doc("other", page=1)]
        sparse = [_doc("dense doc", page=0), _doc("another", page=2)]
        fused = reciprocal_rank_fusion(dense, sparse, top_n=10)
        self.assertEqual(fused[0].page_content, "dense doc")

    def test_same_source_and_page_different_content_not_merged(self):
        # Regression: prefix-based ids (first 40 chars) merged distinct chunks.
        dense = [_doc("x" * 60 + " FIRST UNIQUE TAIL", page=0)]
        sparse = [_doc("x" * 60 + " SECOND UNIQUE TAIL", page=0)]
        fused = reciprocal_rank_fusion(dense, sparse, top_n=10)
        self.assertEqual(len(fused), 2)

    def test_top_n_respected(self):
        docs = [_doc(f"chunk {i}", page=i) for i in range(50)]
        self.assertEqual(len(reciprocal_rank_fusion(docs, [], top_n=7)), 7)


class TestModelChain(unittest.TestCase):
    def test_fallback_models_deduplicated(self):
        self.assertEqual(len(FALLBACK_MODELS), len(set(FALLBACK_MODELS)))
        self.assertTrue(FALLBACK_MODELS)

    def test_rerank_empty_input(self):
        self.assertEqual(rerank_documents("q", []), [])


class TestConversations(unittest.TestCase):
    def test_create_new_conversation_shape(self):
        conv = create_new_conversation()
        self.assertIn("id", conv)
        self.assertIn("title", conv)
        self.assertEqual(conv["messages"], [])

    def test_title_short_query(self):
        self.assertEqual(generate_title_from_query("summarize this"), "Summarize this")

    def test_title_keyword_routing(self):
        self.assertEqual(generate_title_from_query("please summarize the whole document for me"), "Document Summary & Insights")
        self.assertEqual(generate_title_from_query("compare these two documents side by side"), "Comparative Document Analysis")

    def test_title_default_truncation(self):
        title = generate_title_from_query("what exactly is the airspeed velocity of an unladen swallow")
        self.assertTrue(title.endswith("..."))


class TestIndexingMetadata(unittest.TestCase):
    """Chroma rejects None metadata values; non-PDF loaders must omit them.

    Regression: indexing a TXT/MD/DOCX file used to fail with "Expected metadata
    value to be a str, int, float or bool, got None".
    """

    def _extract(self, name: str, content: str):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return extract_document_pages(path, name)

    def _assert_no_none_values(self, docs):
        self.assertTrue(docs)
        for doc in docs:
            for key, value in doc.metadata.items():
                self.assertIsNotNone(value, f"metadata[{key!r}] is None for {doc.metadata.get('source')}")

    def test_markdown_omits_page_and_none_values(self):
        docs = self._extract("notes.md", "# Title\n\nSome body text.")
        self._assert_no_none_values(docs)
        self.assertNotIn("page", docs[0].metadata)
        self.assertEqual(docs[0].metadata["source"], "notes.md")

    def test_txt_omits_page_and_none_values(self):
        docs = self._extract("notes.txt", "Plain body text.")
        self._assert_no_none_values(docs)
        self.assertNotIn("page", docs[0].metadata)

    def test_empty_file_raises_instead_of_indexing_nothing(self):
        with self.assertRaises(ValueError):
            self._extract("blank.md", "   \n\n")

    def test_chroma_client_emits_no_telemetry_errors(self):
        """chromadb 0.4.24 + posthog 7.x are API-incompatible; the guard must hold."""
        import tempfile

        import chromadb
        import services.document_service  # installs the telemetry guard on import

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with self.assertNoLogs("chromadb.telemetry.product.posthog", level="ERROR"):
                client = chromadb.PersistentClient(path=tmp)
                collection = client.create_collection("probe")
                collection.add(ids=["1"], documents=["hello"])


class TestAppRegressions(unittest.TestCase):
    """Guardrails against the specific defects fixed in the audit."""

    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "app.py"), encoding="utf-8") as f:
            cls.app_src = f.read()

    def test_app_div_tags_balanced(self):
        self.assertEqual(self.app_src.count("<div"), self.app_src.count("</div>"))

    def test_app_no_template_placeholders(self):
        self.assertNotIn("[Company]", self.app_src)
        self.assertNotIn("[List key achievements", self.app_src)

    def test_app_has_storage_import(self):
        self.assertIn("clear_session_storage", self.app_src)
        self.assertIn("from services.document_service import", self.app_src)

    def test_app_no_hardcoded_resume_filename(self):
        self.assertNotIn("Aviral_Mishra", self.app_src)

    def test_app_no_seed_conversations(self):
        self.assertNotIn("get_default_seed_conversations", self.app_src)
        self.assertNotIn("conv-1", self.app_src)

    def test_assets_dir_exists(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertTrue(os.path.isdir(os.path.join(root, "assets")))
        self.assertFalse(os.path.isdir(os.path.join(root, "assests")))

    def test_xsrf_enabled(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".streamlit", "config.toml"), encoding="utf-8") as f:
            cfg = f.read()
        self.assertIn("enableXsrfProtection = true", cfg)
        self.assertIn("enableCORS = true", cfg)

    def test_logo_reasonably_small(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo = os.path.join(root, "assets", "ragi-logo.png")
        self.assertLess(os.path.getsize(logo), 100_000)

    def test_citation_page_default_is_whole_document(self):
        # A missing page must render as "Whole document", not "Page 1".
        self.assertIn('"page": doc.metadata.get("page")', self.app_src)
        self.assertNotIn('doc.metadata.get("page", 0)', self.app_src)

    def test_dead_modules_removed(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertFalse(os.path.exists(os.path.join(root, "rag_chain.py")))
        self.assertFalse(os.path.exists(os.path.join(root, "ingest.py")))


class TestQueryRewriteHeuristics(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."}
        ]

    def test_empty_history_never_triggers_rewriting(self):
        self.assertFalse(should_rewrite_query("What about this?", []))

    def test_standalone_query_bypasses_rewriting(self):
        # A complete, standalone query with zero anaphora must bypass rewriting
        standalone = "What was the total net profit recorded in the Q3 2024 financial statement?"
        self.assertFalse(should_rewrite_query(standalone, self.history))

    def test_pronoun_query_triggers_rewriting(self):
        self.assertTrue(should_rewrite_query("What did it say about that?", self.history))
        self.assertTrue(should_rewrite_query("Explain his previous statement", self.history))

    def test_short_elliptical_query_triggers_rewriting(self):
        self.assertTrue(should_rewrite_query("why so?", self.history))
        self.assertTrue(should_rewrite_query("and Q2?", self.history))

    def test_conversational_phrase_triggers_rewriting(self):
        self.assertTrue(should_rewrite_query("tell me more about the architecture", self.history))
        self.assertTrue(should_rewrite_query("what else is mentioned?", self.history))


class TestBM25CacheSafety(unittest.TestCase):
    def test_different_corpuses_cached_independently(self):
        corpus_a = [_doc("quantum entanglement physics particles", source="a.pdf")]
        corpus_b = [_doc("pasta tomato sauce garlic recipe", source="b.pdf")]

        bm25_a = _bm25_for(corpus_a)
        bm25_b = _bm25_for(corpus_b)

        # Different content must yield independent BM25 instances
        self.assertIsNot(bm25_a, bm25_b)

        # Same content must hit the cache and return the exact same instance
        self.assertIs(_bm25_for(corpus_a), bm25_a)
        self.assertIs(_bm25_for(corpus_b), bm25_b)

    def test_empty_corpus(self):
        results = bm25_search("query", [])
        self.assertEqual(results, [])


class TestWorkflowSampling(unittest.TestCase):
    def test_select_representative_chunks_small_document(self):
        # When chunks <= max_count, 100% of chunks must be preserved
        chunks = [_doc(f"chunk {i}") for i in range(10)]
        sampled = select_representative_chunks(chunks, max_count=20)
        self.assertEqual(len(sampled), 10)
        self.assertEqual(sampled, chunks)

    def test_select_representative_chunks_large_document(self):
        # When chunks > max_count, preserves head, progression, and tail up to max_count
        chunks = [_doc(f"chunk {i}") for i in range(100)]
        sampled = select_representative_chunks(chunks, max_count=24)
        self.assertEqual(len(sampled), 24)
        # Head (first 3) must match
        self.assertEqual(sampled[:3], chunks[:3])
        # Tail (last 3) must match
        self.assertEqual(sampled[-3:], chunks[-3:])

    def test_select_representative_chunks_empty(self):
        self.assertEqual(select_representative_chunks([]), [])


class TestSessionStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orig_db_path = session_store.DB_PATH
        self.orig_session_id_file = session_store.SESSION_ID_FILE
        session_store.DB_PATH = os.path.join(self.tmpdir, "test_session.db")
        session_store.SESSION_ID_FILE = os.path.join(self.tmpdir, ".test_session_id")
        session_store.init_db()

    def tearDown(self):
        session_store.DB_PATH = self.orig_db_path
        session_store.SESSION_ID_FILE = self.orig_session_id_file
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_session_id_persistence_and_rotation(self):
        sid1 = session_store.get_or_create_session_id()
        self.assertTrue(sid1)
        # Calling again returns the exact same session id
        sid2 = session_store.get_or_create_session_id()
        self.assertEqual(sid1, sid2)

        # Rotating generates a distinct session id
        sid3 = session_store.rotate_session_id()
        self.assertNotEqual(sid1, sid3)
        self.assertEqual(session_store.get_or_create_session_id(), sid3)

    def test_document_lifecycle(self):
        doc = {
            "id": "doc-1",
            "name": "report.pdf",
            "type": "pdf",
            "size_bytes": 1024,
            "size_str": "1.0 KB",
            "chunks_count": 5,
            "status": "ready",
            "upload_time": "12:00",
            "page_count": 2,
        }
        session_store.save_document("test-sess", doc)
        loaded = session_store.load_documents("test-sess")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["name"], "report.pdf")
        self.assertEqual(loaded[0]["chunks_count"], 5)

        session_store.delete_document("test-sess", "report.pdf")
        self.assertEqual(session_store.load_documents("test-sess"), [])

    def test_chunks_persistence_and_reconstruction(self):
        chunks = [
            Document(page_content="intro chunk", metadata={"source": "report.pdf", "page": 0}),
            Document(page_content="conclusion chunk", metadata={"source": "report.pdf", "page": 1}),
        ]
        session_store.save_chunks("test-sess", "report.pdf", chunks)
        loaded = session_store.load_chunks("test-sess")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].page_content, "intro chunk")
        self.assertEqual(loaded[0].metadata["page"], 0)
        self.assertEqual(loaded[1].page_content, "conclusion chunk")
        self.assertEqual(loaded[1].metadata["source"], "report.pdf")

    def test_conversation_lifecycle(self):
        conv = {
            "id": "conv-1",
            "title": "Quantum Physics Chat",
            "created_at": "12:05",
            "messages": [
                {"role": "user", "content": "What is entanglement?"},
                {"role": "assistant", "content": "Entanglement is..."},
            ],
        }
        session_store.save_conversation("test-sess", conv)
        loaded = session_store.load_conversations("test-sess")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "Quantum Physics Chat")
        self.assertEqual(len(loaded[0]["messages"]), 2)

        session_store.delete_conversation("test-sess", "conv-1")
        self.assertEqual(session_store.load_conversations("test-sess"), [])

    def test_clear_session(self):
        session_store.save_document("test-sess", {"id": "d1", "name": "a.txt"})
        session_store.save_chunks("test-sess", "a.txt", [Document(page_content="text")])
        session_store.save_conversation("test-sess", {"id": "c1", "title": "chat"})

        session_store.clear_session("test-sess")
        self.assertEqual(session_store.load_documents("test-sess"), [])
        self.assertEqual(session_store.load_chunks("test-sess"), [])
        self.assertEqual(session_store.load_conversations("test-sess"), [])

    def test_conversation_numpy_float32_serialization(self):
        import numpy as np
        conv = {
            "id": "conv-np",
            "title": "NumPy Test Chat",
            "created_at": "12:10",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Answer with scores",
                    "sources": [
                        {
                            "source": "doc.pdf",
                            "score": np.float32(0.8924),
                        }
                    ],
                }
            ],
        }
        # This must not raise TypeError: Object of type float32 is not JSON serializable
        session_store.save_conversation("test-sess", conv)
        loaded = session_store.load_conversations("test-sess")
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded[0]["messages"][0]["sources"][0]["score"], 0.8924, places=3)



if __name__ == "__main__":
    unittest.main()
