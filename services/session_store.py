"""Lightweight SQLite-backed persistence for RAGI session state.

Survives browser tab refreshes (F5) by keeping document metadata,
chunk text, and conversation history in a local SQLite database.
The session identity is anchored to a file-based UUID so the same
workspace always resolves to the same ChromaDB collection.

No new runtime dependencies: uses only the Python stdlib sqlite3 module.
"""
import json
import logging
import os
import sqlite3
import uuid
from typing import Any, Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── File paths (relative to the project root) ────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "ragi_session.db")
SESSION_ID_FILE = os.path.join(_ROOT, ".session_id")


# ── Session Identity ──────────────────────────────────────────────────────────
def get_or_create_session_id() -> str:
    """Returns the persistent workspace session ID.

    Reads the ID from ``.session_id`` on every startup so the same
    ChromaDB collection and SQLite rows are reused after a refresh.
    Creates and persists a fresh UUID on the very first run.
    """
    if os.path.exists(SESSION_ID_FILE):
        try:
            with open(SESSION_ID_FILE, "r", encoding="utf-8") as fh:
                sid = fh.read().strip()
                if sid:
                    return sid
        except OSError:
            pass

    sid = str(uuid.uuid4())
    _write_session_id(sid)
    return sid


def rotate_session_id() -> str:
    """Generates and persists a fresh session ID.

    Called after a full workspace reset so the next upload goes into a
    new ChromaDB collection rather than mixing with the purged one.
    """
    new_id = str(uuid.uuid4())
    _write_session_id(new_id)
    return new_id


def _write_session_id(session_id: str) -> None:
    try:
        with open(SESSION_ID_FILE, "w", encoding="utf-8") as fh:
            fh.write(session_id)
    except OSError as exc:
        logger.warning("Could not persist session ID to disk: %s", exc)


# ── Database Initialisation ───────────────────────────────────────────────────
def init_db() -> None:
    """Creates the schema on first run; safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            name          TEXT NOT NULL,
            type          TEXT,
            size_bytes    INTEGER,
            size_str      TEXT,
            chunks_count  INTEGER,
            status        TEXT,
            upload_time   TEXT,
            page_count    INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_docs_session ON documents (session_id);

        CREATE TABLE IF NOT EXISTS chunks (
            rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT NOT NULL,
            doc_name      TEXT NOT NULL,
            page_content  TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks (session_id);

        CREATE TABLE IF NOT EXISTS conversations (
            id            TEXT PRIMARY KEY,
            session_id    TEXT NOT NULL,
            title         TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            messages_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_convs_session ON conversations (session_id);
        """)


def _connect() -> sqlite3.Connection:
    """Opens a connection with row-factory so columns are accessible by name."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Document Records ──────────────────────────────────────────────────────────
def save_document(session_id: str, doc_record: Dict[str, Any]) -> None:
    """Upserts a document metadata record for this session."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (id, session_id, name, type, size_bytes, size_str,
                 chunks_count, status, upload_time, page_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name         = excluded.name,
                type         = excluded.type,
                size_bytes   = excluded.size_bytes,
                size_str     = excluded.size_str,
                chunks_count = excluded.chunks_count,
                status       = excluded.status,
                upload_time  = excluded.upload_time,
                page_count   = excluded.page_count
            """,
            (
                doc_record.get("id", str(uuid.uuid4())),
                session_id,
                doc_record.get("name", ""),
                doc_record.get("type"),
                doc_record.get("size_bytes"),
                doc_record.get("size_str"),
                doc_record.get("chunks_count"),
                doc_record.get("status"),
                doc_record.get("upload_time"),
                doc_record.get("page_count"),
            ),
        )


def load_documents(session_id: str) -> List[Dict[str, Any]]:
    """Returns all document records for this session, ordered by upload time."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE session_id = ? ORDER BY upload_time ASC",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(session_id: str, doc_name: str) -> None:
    """Removes a document record AND its chunks from the store."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM chunks    WHERE session_id = ? AND doc_name = ?",
            (session_id, doc_name),
        )
        conn.execute(
            "DELETE FROM documents WHERE session_id = ? AND name    = ?",
            (session_id, doc_name),
        )


# ── Chunk Records ─────────────────────────────────────────────────────────────
def _json_default(obj: Any) -> Any:
    """Coerces NumPy scalar types, sets, and non-primitive objects to JSON-compatible values."""
    if hasattr(obj, "item"):  # numpy scalar types (float32, int64, etc.)
        return obj.item()
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def save_chunks(session_id: str, doc_name: str, chunks: List[Document]) -> None:
    """Persists chunk text + metadata for a document.

    Replaces any existing rows for the same (session, doc) pair so that
    re-indexing the same file with new content stays consistent.
    """
    with _connect() as conn:
        conn.execute(
            "DELETE FROM chunks WHERE session_id = ? AND doc_name = ?",
            (session_id, doc_name),
        )
        conn.executemany(
            "INSERT INTO chunks (session_id, doc_name, page_content, metadata_json)"
            " VALUES (?, ?, ?, ?)",
            [
                (
                    session_id,
                    doc_name,
                    chunk.page_content,
                    json.dumps(chunk.metadata, default=_json_default),
                )
                for chunk in chunks
            ],
        )


def load_chunks(session_id: str) -> List[Document]:
    """Reconstructs LangChain Document objects from persisted chunk rows."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT page_content, metadata_json FROM chunks"
            " WHERE session_id = ? ORDER BY rowid ASC",
            (session_id,),
        ).fetchall()
    return [
        Document(
            page_content=row["page_content"],
            metadata=json.loads(row["metadata_json"]),
        )
        for row in rows
    ]


# ── Conversation Records ──────────────────────────────────────────────────────
def save_conversation(session_id: str, conv: Dict[str, Any]) -> None:
    """Upserts a conversation (title + full serialised message list)."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, session_id, title, created_at, messages_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title         = excluded.title,
                messages_json = excluded.messages_json
            """,
            (
                conv.get("id", str(uuid.uuid4())),
                session_id,
                conv.get("title", "New Chat"),
                conv.get("created_at", ""),
                json.dumps(conv.get("messages", []), default=_json_default),
            ),
        )


def delete_conversation(session_id: str, conv_id: str) -> None:
    """Removes a single conversation from the store."""
    with _connect() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE session_id = ? AND id = ?",
            (session_id, conv_id),
        )


def load_conversations(session_id: str) -> List[Dict[str, Any]]:
    """Returns all conversations for this session, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
    result = []
    for row in rows:
        record = dict(row)
        record["messages"] = json.loads(record.pop("messages_json", "[]"))
        result.append(record)
    return result


# ── Session Reset ─────────────────────────────────────────────────────────────
def clear_session(session_id: str) -> None:
    """Deletes ALL data rows belonging to the given session."""
    with _connect() as conn:
        conn.execute("DELETE FROM chunks        WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM documents     WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
