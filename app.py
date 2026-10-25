import streamlit as st
import tempfile
import os
import base64
import html
import uuid
from services.document_service import (
    process_and_index_document,
    compute_insights,
    relative_time,
    clear_session_storage,
    delete_document_chunks,
    list_session_collections,
    purge_all_vector_stores,
    filter_documents_by_format,
)
from services.rag_service import (
    get_embeddings,
    stream_rag_answer,
    FALLBACK_MODELS
)
from services.conversation_service import (
    create_new_conversation,
    generate_title_from_query
)
from services.workflow_service import (
    generate_document_summary,
    generate_document_comparison,
    generate_specialized_artifact
)
import services.session_store as session_store
from services.icons import get_icon

# ── Page Configuration ────────────────────────────────────
_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "ragi-logo.png")
_PAGE_ICON = _LOGO_PATH if os.path.exists(_LOGO_PATH) else "📄"

st.set_page_config(
    page_title="RAGI — Document Intelligence",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Helper to Load Local Asset as Base64 ──────────────────
def get_image_base64(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "assets", filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode()}"
    return ""

LOGO_B64 = get_image_base64("ragi-logo.png")
MTN_BL_B64 = get_image_base64("mountain-bottom-left.png")
MTN_TR_B64 = get_image_base64("mountain-top-right.png")


# ── Persistent Store Bootstrap ───────────────────────────
# Init the SQLite schema once per process (idempotent).
session_store.init_db()

# ── Global Session State Setup ────────────────────────────
if "session_id" not in st.session_state:
    # Use the file-anchored workspace ID so F5 returns to the same session.
    st.session_state.session_id = session_store.get_or_create_session_id()

if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Home"

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "active_conv_id" not in st.session_state:
    st.session_state.active_conv_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "deep_search" not in st.session_state:
    st.session_state.deep_search = True

if "show_citations" not in st.session_state:
    st.session_state.show_citations = True

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "smart"

if "selected_scope" not in st.session_state:
    st.session_state.selected_scope = "All Documents"

if "compare_modal" not in st.session_state:
    st.session_state.compare_modal = False

if "generate_modal" not in st.session_state:
    st.session_state.generate_modal = False

if "chat_search_query" not in st.session_state:
    st.session_state.chat_search_query = ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = FALLBACK_MODELS[0]

if "top_k_dense" not in st.session_state:
    st.session_state.top_k_dense = 20

if "top_k_bm25" not in st.session_state:
    st.session_state.top_k_bm25 = 20

if "failed_uploads" not in st.session_state:
    st.session_state.failed_uploads = set()

# ── Restore state from SQLite on first render (survives F5) ──
if "_db_loaded" not in st.session_state:
    _stored_docs = session_store.load_documents(st.session_state.session_id)
    _stored_chunks = session_store.load_chunks(st.session_state.session_id)
    _stored_convs = session_store.load_conversations(st.session_state.session_id)
    if _stored_docs:
        st.session_state.indexed_files = _stored_docs
    if _stored_chunks:
        st.session_state.all_chunks = _stored_chunks
    if _stored_convs:
        st.session_state.conversations = _stored_convs
    st.session_state._db_loaded = True


# ── Shared Session Helpers ────────────────────────────────
def ensure_active_conversation():
    """Lazily creates a conversation on first message if none is active."""
    valid_ids = {c["id"] for c in st.session_state.conversations}
    if st.session_state.active_conv_id not in valid_ids:
        conv = create_new_conversation()
        st.session_state.conversations.insert(0, conv)
        st.session_state.active_conv_id = conv["id"]


def persist_active_messages():
    """Stores the current message list on the active conversation (by copy + SQLite)."""
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_conv_id:
            c["messages"] = list(st.session_state.messages)
            session_store.save_conversation(st.session_state.session_id, c)
            break


def remove_document(doc_name: str):
    """Deletes a document's chunks from session memory, ChromaDB, and SQLite."""
    delete_document_chunks(st.session_state.session_id, doc_name)
    session_store.delete_document(st.session_state.session_id, doc_name)
    st.session_state.indexed_files = [d for d in st.session_state.indexed_files if d["name"] != doc_name]
    st.session_state.all_chunks = [
        c for c in st.session_state.all_chunks if c.metadata.get("source") != doc_name
    ]
    if st.session_state.selected_scope == doc_name:
        st.session_state.selected_scope = "All Documents"


def index_uploaded_files(uploaded_files) -> bool:
    """Indexes new uploads; replaces same-name files whose content changed;
    remembers failures so they are not retried on every rerun.

    Returns True when the index changed."""
    existing = {d["name"]: d.get("size_bytes") for d in st.session_state.indexed_files}
    seen = set()
    changed = False

    for file_obj in uploaded_files:
        name = file_obj.name
        if name in seen or name in st.session_state.failed_uploads:
            continue
        seen.add(name)

        data = file_obj.read()
        if name in existing and existing[name] == len(data):
            continue  # unchanged file already indexed

        file_ext = os.path.splitext(name)[1]
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            if name in existing:
                # Same filename, new content: replace the old chunks.
                remove_document(name)

            doc_meta, chunks = process_and_index_document(
                tmp_path, name, st.session_state.session_id, get_embeddings()
            )
            st.session_state.all_chunks.extend(chunks)
            st.session_state.indexed_files.append(doc_meta)
            # Persist to SQLite so an F5 refresh restores this document.
            session_store.save_document(st.session_state.session_id, doc_meta)
            session_store.save_chunks(st.session_state.session_id, name, chunks)
            changed = True
        except Exception as e:
            st.session_state.failed_uploads.add(name)
            st.error(f"Could not index **{html.escape(name)}**: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return changed


# ── Precision Editorial Design System CSS ─────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Cormorant+Garamond:ital,wght@0,600;0,700;1,400;1,600&display=swap');

    :root {{
        --paper: #FAF7F2;
        --paper-card: #FFFFFF;
        --sidebar-bg: #F5EFE6;
        --ink-primary: #2D2319;
        --ink-secondary: #7A695C;
        --ink-muted: #8C7A6B;
        --espresso: #4A3326;
        --espresso-hover: #362217;
        --border-color: #E8DFD0;
        --border-subtle: #F2ECE1;
        --sand-active: #E8DFD0;
    }}

    /* ── Global Body Canvas ─────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: var(--paper) !important;
        color: var(--ink-primary) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }}

    /* Hide Streamlit header chrome */
    [data-testid="stHeader"] {{
        background: transparent !important;
        height: 0px !important;
    }}

    /* ═══ LEFT SIDEBAR — Fixed shell, only chats scroll ═══ */
    [data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
        padding: 0 !important;
    }}

    /* The inner wrapper must be a non-scrolling flex column */
    [data-testid="stSidebar"] > div:first-child {{
        padding: 0 !important;
        margin: 0 !important;
        height: 100vh !important;
        overflow: hidden !important;          /* ← NO overall scroll */
        display: flex !important;
        flex-direction: column !important;
    }}

    /* Make sidebar's stVerticalBlockBorderWrapper fill height */
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
        flex: 1 !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        flex: 1 !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        overflow: hidden !important;
        gap: 0 !important;
    }}

    /* Mountain watermark inside sidebar */
    {'section[data-testid="stSidebar"]::after { content: ""; position: fixed; bottom: 0px; left: 0; width: 220px; height: 130px; background-image: url(' + MTN_BL_B64 + '); background-size: contain; background-repeat: no-repeat; background-position: bottom left; opacity: 0.35; pointer-events: none; z-index: 0; }' if MTN_BL_B64 else ''}

    /* ── Sidebar Top Spacing: RAGI flush to top-left corner ── */
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"],
    [data-testid="stSidebarHeader"] {{
        padding-top: 6px !important;
        padding-bottom: 0px !important;
        height: auto !important;
        min-height: 0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarContent"] {{
        padding-top: 6px !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
    }}

    /* ── Sidebar sub-sections ──────────────────────────── */
    /* Fixed top brand area */
    .sidebar-fixed-top {{
        flex-shrink: 0;
        padding: 0px 4px 6px 4px !important;
    }}
    .ragi-top-brand {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }}
    .ragi-logo-lockup {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .ragi-brand-logo-img {{
        width: 32px;
        height: 32px;
        object-fit: contain;
    }}
    .ragi-brand-name {{
        font-family: 'Cinzel', serif;
        font-size: 19px;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: var(--espresso-hover);
        margin: 0;
        line-height: 1.1;
    }}
    .ragi-brand-sub {{
        font-size: 10px;
        color: var(--ink-muted);
        margin: 0;
        letter-spacing: 0.02em;
    }}

    /* Fixed nav area */
    .sidebar-fixed-nav {{
        flex-shrink: 0;
        padding: 0 14px 4px 14px;
    }}

    /* Navigation Radio: clean vector SVG icons */
    div[data-testid="stRadio"] > div {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}
    div[data-testid="stRadio"] label {{
        background: transparent !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        color: #524235 !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        display: flex !important;
        align-items: center !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        background: #EAE1D2 !important;
    }}
    div[data-testid="stRadio"] label:nth-of-type(1) div[data-testid="stMarkdownContainer"] p::before {{
        content: "";
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 8px;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%23524235" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>');
    }}
    div[data-testid="stRadio"] label:nth-of-type(2) div[data-testid="stMarkdownContainer"] p::before {{
        content: "";
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 8px;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%23524235" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>');
    }}
    div[data-testid="stRadio"] label:nth-of-type(3) div[data-testid="stMarkdownContainer"] p::before {{
        content: "";
        display: inline-block;
        width: 15px;
        height: 15px;
        margin-right: 8px;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%23524235" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>');
    }}
    /* Keep radios focusable for keyboard users, just visually hidden */
    div[data-testid="stRadio"] input[type="radio"] {{
        position: absolute !important;
        opacity: 0 !important;
        width: 1px !important;
        height: 1px !important;
    }}
    div[data-testid="stRadio"] label:focus-within {{
        outline: 2px solid var(--espresso) !important;
        outline-offset: 1px !important;
    }}

    /* Chats section label */
    .sidebar-chats-label {{
        flex-shrink: 0;
        padding: 8px 14px 4px 18px;
        font-size: 11.5px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--ink-muted);
    }}

    /* SCROLLABLE recent-chats area — a real container node (marker inside it) */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> .element-container .chats-scroll-marker) {{
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overscroll-behavior: contain;
        padding: 0 4px 8px 4px;
        position: relative;
        z-index: 1;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> .element-container .chats-scroll-marker)::-webkit-scrollbar {{
        width: 4px;
    }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> .element-container .chats-scroll-marker)::-webkit-scrollbar-thumb {{
        background: #D8CEBE;
        border-radius: 4px;
    }}

    /* Layout markers only identify structure — hide the element carrying them */
    .element-container:has(.chats-scroll-marker),
    .element-container:has(.chat-scroll-marker),
    .element-container:has(.asst-card-marker),
    .element-container:has(.right-panel-marker),
    .element-container:has(.scope-bar-marker) {{
        display: none !important;
    }}

    /* Fixed bottom profile card: pinned firmly to the bottom */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.sidebar-fixed-bottom),
    [data-testid="stSidebar"] .element-container:has(.sidebar-fixed-bottom),
    .sidebar-fixed-bottom {{
        margin-top: auto !important;
        flex-shrink: 0 !important;
    }}
    .sidebar-fixed-bottom {{
        padding: 8px 10px 14px 10px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }}
    .profile-dock-card {{
        background-color: #EAE1D2;
        border-radius: 12px;
        padding: 10px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        position: relative;
        z-index: 2;
        box-shadow: 0 1px 3px rgba(45, 35, 25, 0.04);
    }}
    .profile-avatar-circle {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background-color: #7A6354;
        color: #FAF7F2;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12.5px;
        font-weight: 600;
    }}

    /* ═══ APP SHELL — fixed viewport, internal scroll regions only ═══ */
    html, body, #root, .stApp, [data-testid="stAppViewContainer"] {{
        height: 100vh;
        overflow: hidden !important;
    }}
    section.main {{
        height: 100vh !important;
        overflow: hidden !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 0 !important;
    }}

    /* The middle column is its own scroll container. On the chat view its
       content fits exactly, so only the chat box actually scrolls. */
    .block-container {{
        flex: 1 1 auto !important;
        min-height: 0 !important;
        height: auto !important;
        max-height: none !important;
        box-sizing: border-box !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0 !important;
        padding-left: 1.2rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
        width: 100% !important;
        position: relative;
        z-index: 1;
    }}

    /* Stretch chain: every wrapper between the viewport and the content fills
       the height instead of growing past it (non-chat views keep scrolling
       in .block-container itself). The unnamed div inside every
       stVerticalBlockBorderWrapper must stretch too, or the chain breaks and
       content overflows the viewport. */
    .block-container [data-testid="stVerticalBlockBorderWrapper"],
    .block-container [data-testid="stVerticalBlockBorderWrapper"] > div,
    .block-container [data-testid="stVerticalBlock"] {{
        flex: 1 1 auto !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }}
    /* Same for the layers between the fixed sidebar shell and its blocks */
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div {{
        flex: 1 1 auto !important;
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
    }}
    /* Streamlit gives every column-row flex-grow:1, which makes header and
       scope rows swallow free space. Rows keep their natural height; only the
       chat canvas below is allowed to grow. */
    [data-testid="stHorizontalBlock"] {{
        flex-grow: 0 !important;
    }}
    /* Home's top row (chat column + right rail) and its two columns */
    .block-container [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) {{
        flex: 1 1 auto !important;
        min-height: 0 !important;
    }}
    /* Home's top row (chat column + right rail): main chat column expands, right rail is fixed & clean */
    .block-container [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) > [data-testid="column"]:first-child {{
        flex: 1 1 0% !important;
        min-width: 0 !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }}
    .block-container [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) > [data-testid="column"]:last-child {{
        flex: 0 0 340px !important;
        width: 340px !important;
        max-width: 360px !important;
        min-width: 290px !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
    }}

    /* ── Chat canvas: the ONLY scrolling region in the middle column.
          The header (title + cards) sits above it and the scope bar + input
          sit below it, so both stay pinned while messages scroll. ── */
    .block-container [data-testid="stVerticalBlock"]:has(> .element-container .chat-scroll-marker) {{
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overscroll-behavior: contain;
        padding-bottom: 24px !important;
    }}
    .block-container [data-testid="stVerticalBlock"]:has(> .element-container .chat-scroll-marker)::-webkit-scrollbar {{
        width: 5px;
    }}
    .block-container [data-testid="stVerticalBlock"]:has(> .element-container .chat-scroll-marker)::-webkit-scrollbar-thumb {{
        background: #D5C8B5;
        border-radius: 4px;
    }}

    /* ── Docked scope bar, immediately above the fixed input ── */
    .element-container:has(.scope-bar-marker) + [data-testid="stHorizontalBlock"] {{
        flex: 0 0 auto !important;
        background: rgba(250, 247, 242, 0.98) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 14px;
        padding: 6px 10px 2px;
        margin-top: 4px;
        position: relative;
        z-index: 5;
    }}

    /* ═══ FIXED BOTTOM CHAT INPUT (ChatGPT Style) ═══ */
    [data-testid="stChatInput"] {{
        flex-shrink: 0 !important;
        margin-top: 6px !important;
        margin-bottom: 6px !important;
        background: #FFFFFF !important;
        border: 1.5px solid #E2D7C7 !important;
        border-radius: 18px !important;
        box-shadow: 0 4px 20px rgba(45, 35, 25, 0.06) !important;
        transition: all 0.2s ease !important;
        position: relative;
        z-index: 5;
    }}
    [data-testid="stChatInput"] > div {{
        background: transparent !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        border-color: #4A3326 !important;
        box-shadow: 0 4px 24px rgba(74, 51, 38, 0.12) !important;
    }}



    /* Toggles & scope selector: keep text on one line, never wrap characters vertically */
    [data-testid="stToggle"] {{
        white-space: nowrap !important;
        min-width: fit-content !important;
    }}
    [data-testid="stToggle"] label,
    [data-testid="stToggle"] label span,
    [data-testid="stToggle"] p {{
        white-space: nowrap !important;
        word-break: keep-all !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #524235 !important;
    }}

    /* Action buttons: sleek vector SVGs */
    div[data-testid="stHorizontalBlock"] button[key^="act_"] {{
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        border-radius: 12px !important;
        padding: 8px 10px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stHorizontalBlock"] button[key="act_sum_btn"] div p::before {{
        content: "";
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 6px;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%232D2319" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>');
    }}
    div[data-testid="stHorizontalBlock"] button[key="act_find_btn"] div p::before {{
        content: "";
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 6px;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%232D2319" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>');
    }}
    div[data-testid="stHorizontalBlock"] button[key="act_comp_btn"] div p::before {{
        content: "";
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 6px;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%232D2319" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/></svg>');
    }}
    div[data-testid="stHorizontalBlock"] button[key="act_gen_btn"] div p::before {{
        content: "";
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 6px;
        vertical-align: -2px;
        background-repeat: no-repeat;
        background-size: contain;
        background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="%232D2319" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>');
    }}

    .main-editorial-title {{
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: clamp(24px, 2.6vw, 36px) !important;
        font-weight: 700;
        color: #2A1D15;
        text-align: center;
        margin: 0 0 2px 0;
        letter-spacing: -0.01em;
        line-height: 1.15;
    }}
    .main-editorial-sub {{
        font-size: clamp(12px, 1.1vw, 13.5px) !important;
        color: var(--ink-secondary);
        text-align: center;
        margin-bottom: 14px;
    }}

    /* ── Action Cards Polish ────────────────────────────── */
    .action-card-box {{
        background: #FFFFFF;
        border: 1px solid #EAE1D2;
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 1px 4px rgba(45, 35, 25, 0.02);
        display: flex;
        align-items: center;
        gap: 12px;
        height: 72px;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .action-card-box:hover {{
        border-color: #CFC1AF;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(45, 35, 25, 0.05);
    }}
    .action-icon-circle {{
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }}

    /* Chat bubble styling */
    .user-msg-container {{
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        gap: 10px;
        margin: 16px 0;
    }}
    .user-msg-bubble {{
        background-color: #EDE5D8;
        border: 1px solid #DDD3C3;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        color: var(--ink-primary);
        font-size: 14px;
        max-width: 78%;
        line-height: 1.55;
    }}
    .user-monogram-avatar {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: #A68B74;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 600;
        flex-shrink: 0;
    }}

    /* Assistant Research Card */
    .assistant-card-row {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 20px;
    }}
    .assistant-flower-avatar {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background-color: var(--espresso);
        color: #FAF7F2;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        flex-shrink: 0;
        margin-top: 4px;
    }}
    .assistant-white-card {{
        background-color: var(--paper-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 2px 10px rgba(45, 35, 25, 0.03);
        flex-grow: 1;
    }}

    /* ═══ RIGHT RAIL — fixed viewport height, content scrolls inside ═══ */
    [data-testid="column"]:has(.right-panel-marker) {{
        min-height: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        border-left: 1px solid var(--border-color) !important;
        padding-left: 16px !important;
        padding-right: 4px !important;
        margin-left: 2px !important;
    }}
    [data-testid="column"]:has(.right-panel-marker) > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {{
        flex: 1 1 auto !important;
        min-height: 0 !important;
        overflow-y: auto !important;
        overscroll-behavior: contain;
    }}
    [data-testid="column"]:has(.right-panel-marker) > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"]::-webkit-scrollbar {{
        width: 4px;
    }}
    [data-testid="column"]:has(.right-panel-marker) > [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {{
        background: #D5C8B5;
        border-radius: 4px;
    }}

    /* ── Assistant message card: one container holds header, answer,
          and citations together (they used to render as split siblings) ── */
    .block-container [data-testid="stVerticalBlock"]:has(> .element-container .asst-card-marker) {{
        flex: 0 1 auto !important;
        min-height: auto !important;
        background-color: var(--paper-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(45, 35, 25, 0.03);
        margin-bottom: 14px;
    }}
    .doc-source-header-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding-bottom: 10px;
        margin-bottom: 10px;
        border-bottom: 1px solid var(--border-subtle);
    }}

    .studio-panel-box {{
        background-color: var(--paper-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(45, 35, 25, 0.02);
    }}
    .studio-panel-title {{
        font-size: 13px;
        font-weight: 700;
        color: var(--ink-primary);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    .insights-2x2-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }}
    .insights-cell {{
        background-color: var(--paper);
        border: 1px solid #F0EAE0;
        border-radius: 10px;
        padding: 10px 12px;
    }}
    .insights-big-num {{
        font-size: 17px;
        font-weight: 700;
        color: var(--espresso-hover);
        margin: 0;
    }}
    .insights-small-lbl {{
        font-size: 11px;
        color: var(--ink-muted);
        margin: 0;
    }}

    .indexed-pill-green {{
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }}

    /* Floating Input Toolbar */
    .input-control-toolbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 4px;
        padding: 0 6px;
    }}

    /* ── Responsive: keep the shell intact on narrow viewports ── */
    @media (max-width: 1080px) {{
        .block-container [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) > [data-testid="column"]:last-child {{
            flex: 0 0 270px !important;
            width: 270px !important;
        }}
    }}
    @media (max-width: 860px) {{
        .block-container [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) {{
            flex-direction: column !important;
            height: auto !important;
        }}
        .block-container [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) > [data-testid="column"]:first-child,
        .block-container [data-testid="stHorizontalBlock"]:has(.chat-scroll-marker) > [data-testid="column"]:last-child {{
            width: 100% !important;
            max-width: 100% !important;
            flex: 1 1 auto !important;
            height: auto !important;
            border-left: none !important;
        }}
        [data-testid="column"]:has(.right-panel-marker) {{
            padding-left: 0 !important;
            border-top: 1px solid var(--border-color) !important;
            padding-top: 14px !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# ── Left Fixed Navigation Sidebar ────────────────────────
with st.sidebar:
    # ─── FIXED TOP: Brand Lockup ───
    logo_img_tag = f'<img src="{LOGO_B64}" class="ragi-brand-logo-img">' if LOGO_B64 else f'<div style="width:32px;height:32px;display:flex;align-items:center;justify-content:center;background:#4A3326;border-radius:8px;">{get_icon("sparkles", size=18, color="#FAF7F2")}</div>'
    st.markdown(f"""
    <div class="sidebar-fixed-top">
        <div class="ragi-top-brand">
            <div class="ragi-logo-lockup">
                {logo_img_tag}
                <div>
                    <div class="ragi-brand-name">RAGI</div>
                    <div class="ragi-brand-sub">Your Knowledge, Amplified</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── FIXED: + New Chat Button ───
    if st.button("＋  New Chat", use_container_width=True):
        persist_active_messages()
        # Prune chats that were opened but never used so the list stays clean.
        st.session_state.conversations = [c for c in st.session_state.conversations if c["messages"]]
        new_conv = create_new_conversation()
        st.session_state.conversations.insert(0, new_conv)
        st.session_state.active_conv_id = new_conv["id"]
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.active_nav = "Home"
        st.rerun()

    # ─── FIXED: Navigation Items ───
    nav_icons = {"Home": "Home", "Knowledge Base": "Knowledge Base", "Settings": "Settings"}
    nav_options = ["Home", "Knowledge Base", "Settings"]
    current_nav = "Knowledge Base" if st.session_state.active_nav in ["Documents", "Library", "Knowledge Base"] else st.session_state.active_nav
    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        format_func=lambda x: nav_icons.get(x, x),
        index=nav_options.index(current_nav) if current_nav in nav_options else 0,
        label_visibility="collapsed"
    )
    if selected_nav != st.session_state.active_nav:
        st.session_state.active_nav = selected_nav
        st.rerun()

    # ─── FIXED LABEL: Recent Chats ───
    st.markdown('<div class="sidebar-chats-label">Recent Chats</div>', unsafe_allow_html=True)

    # ─── Search box for recent chats ───
    chat_search = st.text_input("Search chats", value=st.session_state.chat_search_query, placeholder="Search recent chats...", label_visibility="collapsed")
    st.session_state.chat_search_query = chat_search

    # ─── SCROLLABLE: Recent Chats List ───
    filtered_convs = [
        c for c in st.session_state.conversations
        if not chat_search or chat_search.lower() in c["title"].lower()
    ]

    # Scroll region = a real Streamlit container (one DOM node the CSS can size
    # and scroll). A markdown div opened here and closed after the widgets is
    # auto-closed by the browser, so it never actually wraps them.
    chats_scroll_box = st.container()
    with chats_scroll_box:
        st.markdown('<div class="chats-scroll-marker"></div>', unsafe_allow_html=True)
        if not filtered_convs:
            st.caption("No chats yet — ask a question to start one.")
        else:
            for conv in filtered_convs:
                title = conv['title']
                label = f"{title[:22]}..." if len(title) > 22 else title
                open_col, del_col = st.columns([1, 0.22])
                with open_col:
                    if st.button(label, key=f"conv_{conv['id']}", use_container_width=True,
                                 help=f"Created {conv.get('created_at', '')}"):
                        persist_active_messages()
                        st.session_state.active_conv_id = conv["id"]
                        st.session_state.messages = list(conv.get("messages", []))
                        st.session_state.active_nav = "Home"
                        st.rerun()
                with del_col:
                    if st.button("✕", key=f"delconv_{conv['id']}", use_container_width=True,
                                 help="Delete this chat"):
                        session_store.delete_conversation(st.session_state.session_id, conv["id"])
                        st.session_state.conversations = [
                            c for c in st.session_state.conversations if c["id"] != conv["id"]
                        ]
                        if st.session_state.active_conv_id == conv["id"]:
                            st.session_state.active_conv_id = None
                            st.session_state.messages = []
                        st.rerun()

    # ─── FIXED BOTTOM: User Profile Card ───
    st.markdown("""
    <div class="sidebar-fixed-bottom">
        <div class="profile-dock-card">
            <div class="profile-avatar-circle">G</div>
            <div style="flex-grow:1;">
                <div style="font-size:13px; font-weight:600; color:#2D2319;">Guest User</div>
                <div style="font-size:11px; color:#8C7A6B;">Local session • no account needed</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# VIEW 1: HOME — THE 3-COLUMN RAG RESEARCH WORKSPACE
# ══════════════════════════════════════════════════════════
if st.session_state.active_nav == "Home":
    main_col, right_col = st.columns([0.72, 0.28], gap="medium")

    # ── Right Panel — Mountain at top, then scrollable content ────
    with right_col:
        # Marker: the CSS turns this column into a fixed-height right rail whose
        # content scrolls inside (the old wrapper div was an empty, auto-closed
        # element and never contained this content).
        st.markdown('<div class="right-panel-marker"></div>', unsafe_allow_html=True)

        # ── Mountain image at the very top of the right panel ──
        if MTN_TR_B64:
            st.markdown(f'''
            <div style="width:100%; margin:0 0 12px 0; border-radius:12px; overflow:hidden; line-height:0;">
                <img src="{MTN_TR_B64}" style="width:100%; max-height:130px; object-fit:cover; border-radius:12px; opacity:0.92;">
            </div>
            ''', unsafe_allow_html=True)

        insights = compute_insights(st.session_state.indexed_files, st.session_state.all_chunks)

        # 1. Documents Header & File Dropzone
        st.markdown(f"""
        <div class="studio-panel-title">
            <span style="display:flex; align-items:center; gap:8px;">{get_icon('folder', size=16, color='#4A3326')} Documents</span>
            <span style="font-size:11.5px; color:#8C7A6B; font-weight:500;">{insights['doc_count']} file{'s' if insights['doc_count'] != 1 else ''}</span>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Drag & Drop PDF, DOCX, TXT or MD files (Max 200MB)"
        )

        if st.session_state.failed_uploads:
            st.warning("Some uploads failed: " + ", ".join(sorted(st.session_state.failed_uploads)))
            if st.button("Retry failed uploads", key="retry_failed_home"):
                st.session_state.failed_uploads = set()
                st.rerun()

        if uploaded_files:
            with st.spinner("Processing & indexing document(s)..."):
                changed = index_uploaded_files(uploaded_files)
            if changed:
                st.rerun()

        # Render Active Document Cards
        for doc_item in st.session_state.indexed_files:
            safe_doc_name = html.escape(doc_item['name'])
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between; background:#FFFFFF; border:1px solid #EBE3D7; border-radius:10px; padding:10px 12px; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="display:flex; align-items:center;">{get_icon('file-text', size=16, color='#7A695C')}</span>
                    <div>
                        <div style="font-size:12.5px; font-weight:600; color:#2D2319; word-break:break-all;">{safe_doc_name}</div>
                        <div style="font-size:11px; color:#8C7A6B;">{doc_item['size_str']} • {doc_item['chunks_count']} chunks • indexed {relative_time(doc_item.get('upload_time', ''))}</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span class="indexed-pill-green">{doc_item.get('status', 'Indexed')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Dynamic Insights 2x2 Grid (100% Dynamic)
        st.markdown(f"""
        <div class="studio-panel-box" style="margin-top:12px;">
            <div class="studio-panel-title">
                <span style="display:flex; align-items:center; gap:8px;">{get_icon('bar-chart', size=16, color='#4A3326')} Document Insights</span>
            </div>
            <div class="insights-2x2-grid">
                <div class="insights-cell">
                    <div class="insights-big-num">{insights['total_chunks']}</div>
                    <div class="insights-small-lbl">Total Chunks</div>
                </div>
                <div class="insights-cell">
                    <div class="insights-big-num">{insights['file_size']}</div>
                    <div class="insights-small-lbl">File Size</div>
                </div>
                <div class="insights-cell">
                    <div class="insights-big-num">{insights['doc_count']}</div>
                    <div class="insights-small-lbl">Documents</div>
                </div>
                <div class="insights-cell">
                    <div class="insights-big-num">{insights['last_indexed']}</div>
                    <div class="insights-small-lbl">Last Indexed</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Starter Prompts (doc-agnostic)
        st.markdown(f"""
        <div class="studio-panel-box">
            <div class="studio-panel-title">
                <span style="display:flex; align-items:center; gap:8px;">{get_icon('sparkles', size=16, color='#4A3326')} Starter Prompts</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        suggested_list = [
            "Summarize the main points of my documents",
            "List the key facts, terms, or concepts",
            "Extract any tables or structured data",
            "What are the main conclusions or recommendations?",
            "Turn the content into study notes"
        ]

        for q_text in suggested_list:
            if st.button(q_text, key=f"sug_btn_{q_text}", use_container_width=True):
                st.session_state.pending_prompt = q_text
                st.rerun()

        # Reset & Purge Option
        if st.session_state.indexed_files:
            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            if st.button("Reset Workspace & Clear Documents", use_container_width=True):
                clear_session_storage(st.session_state.session_id)
                session_store.clear_session(st.session_state.session_id)
                st.session_state.messages = []
                st.session_state.indexed_files = []
                st.session_state.all_chunks = []
                st.session_state.failed_uploads = set()
                st.session_state.selected_scope = "All Documents"
                st.session_state.conversations = []
                st.session_state.active_conv_id = None
                st.session_state.session_id = session_store.rotate_session_id()
                st.session_state._db_loaded = True
                st.rerun()



    # ── Center Main Workspace ────────────────────────────
    with main_col:
        st.markdown("""
        <div class="main-editorial-title">RAG Document Intelligence</div>
        <div class="main-editorial-sub">Ask questions, extract insights, and get accurate answers from your documents.</div>
        """, unsafe_allow_html=True)

        # 4 Polished Quick Action Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Summarize", key="act_sum_btn", use_container_width=True, help="Create an executive summary of your documents"):
                if not st.session_state.indexed_files:
                    st.warning("Upload a document first.")
                else:
                    active_doc = st.session_state.indexed_files[0]["name"]
                    ensure_active_conversation()
                    st.session_state.messages.append(
                        {"role": "user", "content": f"Create an executive summary of '{active_doc}'."}
                    )
                    try:
                        summary_gen, _ = generate_document_summary(
                            st.session_state.all_chunks, active_doc,
                            model_name=st.session_state.selected_model
                        )
                        with st.spinner("Generating summary..."):
                            result = st.write_stream(summary_gen)
                    except Exception as e:
                        result = f"⚠️ Could not generate summary: {e}"
                    st.session_state.messages.append({
                        "id": str(uuid.uuid4()), "role": "assistant", "content": result,
                        "primary_doc": active_doc, "doc_stats": "Executive summary", "sources": []
                    })
                    persist_active_messages()
                    st.rerun()
        with c2:
            if st.button("Find Info", key="act_find_btn", use_container_width=True, help="Ask specific questions & extract key facts"):
                st.session_state.pending_prompt = "Extract the most important facts, figures, and key terms from my documents."
                st.rerun()
        with c3:
            if st.button("Compare", key="act_comp_btn", use_container_width=True, help="Analyze & compare across documents"):
                st.session_state.compare_modal = not st.session_state.compare_modal
                st.rerun()
        with c4:
            if st.button("Generate", key="act_gen_btn", use_container_width=True, help="Create study notes, reports & artifacts"):
                st.session_state.generate_modal = not st.session_state.generate_modal
                st.rerun()

        # Compare Studio Drawer
        if st.session_state.compare_modal:
            with st.expander("Document Comparison Studio", expanded=True):
                doc_names = [d["name"] for d in st.session_state.indexed_files]
                if len(doc_names) >= 2:
                    cmp1, cmp2 = st.columns(2)
                    with cmp1:
                        sel_a = st.selectbox("Select Document A", doc_names, index=0)
                    with cmp2:
                        sel_b = st.selectbox("Select Document B", doc_names, index=1)
                    if st.button("Run Comparison Analysis", use_container_width=True):
                        ensure_active_conversation()
                        st.session_state.messages.append(
                            {"role": "user", "content": f"Compare '{sel_a}' with '{sel_b}'."}
                        )
                        st.session_state.compare_modal = False
                        try:
                            cmp_gen = generate_document_comparison(
                                sel_a, sel_b, st.session_state.all_chunks,
                                model_name=st.session_state.selected_model
                            )
                            with st.spinner("Running comparative analysis..."):
                                cmp_result = st.write_stream(cmp_gen)
                        except Exception as e:
                            cmp_result = f"⚠️ Comparison failed: {e}"
                        st.session_state.messages.append({
                            "id": str(uuid.uuid4()), "role": "assistant", "content": cmp_result,
                            "primary_doc": f"{sel_a} vs {sel_b}",
                            "doc_stats": "Comparative analysis", "sources": []
                        })
                        persist_active_messages()
                        st.rerun()
                else:
                    st.info("Upload at least two documents in the right panel to run comparative analysis.")

        # Generate Studio Drawer
        if st.session_state.generate_modal:
            with st.expander("Generate Notes & Reports Studio", expanded=True):
                doc_names = [d["name"] for d in st.session_state.indexed_files]
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    gen_artifact = st.selectbox("Artifact Type", ["Study Notes", "Executive Summary", "Technical Report", "FAQ", "Flashcards"])
                with gcol2:
                    gen_doc = st.selectbox("Source Document", doc_names if doc_names else ["Document"])
                if st.button("Generate Artifact", use_container_width=True, disabled=not st.session_state.indexed_files):
                    ensure_active_conversation()
                    st.session_state.messages.append(
                        {"role": "user", "content": f"Generate {gen_artifact} for '{gen_doc}'."}
                    )
                    st.session_state.generate_modal = False
                    try:
                        art_gen = generate_specialized_artifact(
                            gen_artifact, gen_doc, st.session_state.all_chunks,
                            model_name=st.session_state.selected_model
                        )
                        with st.spinner(f"Generating {gen_artifact}..."):
                            art_result = st.write_stream(art_gen)
                    except Exception as e:
                        art_result = f"⚠️ Generation failed: {e}"
                    st.session_state.messages.append({
                        "id": str(uuid.uuid4()), "role": "assistant", "content": art_result,
                        "primary_doc": gen_doc, "doc_stats": gen_artifact, "sources": []
                    })
                    persist_active_messages()
                    st.rerun()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # ── Chat Canvas ──────────────────────────────────
        # One container = one DOM node: the header stays pinned above it, the
        # scope bar + input stay pinned below it, and only this box scrolls.
        chat_box = st.container()
        with chat_box:
            st.markdown('<div class="chat-scroll-marker"></div>', unsafe_allow_html=True)

            if not st.session_state.messages:
                # Honest empty state — no fabricated messages or placeholder answers.
                doc_count = len(st.session_state.indexed_files)
                if doc_count:
                    plural = "s" if doc_count != 1 else ""
                    empty_body = (
                        f"<div style=\"font-weight:700; font-size:15px; margin-bottom:6px\">Ready to answer</div>"
                        f"<div>{doc_count} document{plural} indexed ({len(st.session_state.all_chunks)} chunks). "
                        "Ask a question below, or use a starter prompt to get going.</div>"
                    )
                else:
                    empty_body = (
                        "<div style=\"font-weight:700; font-size:15px; margin-bottom:6px\">No documents yet</div>"
                        "<div>Upload a PDF, DOCX, TXT, or MD file in the <strong>Documents</strong> "
                        "panel on the right — then ask anything about its content.</div>"
                    )
                st.markdown(f"""
                <div class="assistant-card-row">
                    <div class="assistant-flower-avatar">{get_icon('sparkles', size=15, color='#FAF7F2')}</div>
                    <div class="assistant-white-card">
                        <div style="font-size:14px; line-height:1.65; color:#362217;">{empty_body}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="user-msg-container">
                            <div class="user-msg-bubble">{html.escape(msg['content'])}</div>
                            <div class="user-monogram-avatar">A</div>
                        </div>
                        """, unsafe_allow_html=True)

                    elif msg["role"] == "assistant":
                        doc_name = html.escape(msg.get("primary_doc", st.session_state.indexed_files[0]["name"] if st.session_state.indexed_files else "Document"))
                        doc_stats = html.escape(msg.get("doc_stats", "Indexed document"))

                        # The card is a container so header, answer and citations
                        # render INSIDE it. The old code split one card across
                        # three st.markdown calls; the browser auto-closed those
                        # divs, so the answer text ended up outside the card.
                        asst_card = st.container()
                        with asst_card:
                            st.markdown('<div class="asst-card-marker"></div>', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="doc-source-header-row">
                                <div style="display:flex; align-items:center; gap:10px;">
                                    <div class="assistant-flower-avatar" style="margin-top:0;">{get_icon('sparkles', size=15, color='#FAF7F2')}</div>
                                    <div>
                                        <div style="font-size:13.5px; font-weight:600; color:#2D2319; display:flex; align-items:center; gap:6px;">{get_icon('file-text', size=14, color='#4A3326')} {doc_name}</div>
                                        <div style="font-size:11.5px; color:#8C7A6B;">{doc_stats}</div>
                                    </div>
                                </div>
                                <div style="font-size:11.5px; color:#2E7D32; font-weight:600; display:flex; align-items:center; gap:4px; background:#E8F5E9; padding:2px 8px; border-radius:12px;">
                                    {get_icon('shield-check', size=13, color='#2E7D32')}
                                    <span>Verified Response</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown(msg["content"])

                            if "sources" in msg and msg["sources"] and st.session_state.show_citations:
                                with st.expander("Verified Source Citations", expanded=False):
                                    for src_item in msg["sources"]:
                                        doc_src = html.escape(src_item.get("source", "Document"))
                                        page_num = src_item.get("page")
                                        loc_label = f"Page {page_num + 1}" if isinstance(page_num, int) else "Whole document"
                                        snippet = src_item.get("content", "")
                                        st.markdown(f"**{get_icon('file-text', size=14, color='#7A695C')} {doc_src} — {loc_label}**", unsafe_allow_html=True)
                                        st.markdown(f"> *\"{snippet}...\"*")
                                        st.divider()

            # Live exchange renders here — inside the scroll box, above the
            # docked scope bar and input.
            stream_slot = st.empty()

        # ── Scope & Filter Controls (docked right above the fixed chat input) ──
        # No wrapper div: this row is styled as the docked bar by CSS (marker
        # marks where the bar starts; wrapper divs across elements get closed
        # by the browser and never contain the widgets).
        scope_options = ["All Documents"] + [d["name"] for d in st.session_state.indexed_files]
        st.markdown('<div class="scope-bar-marker"></div>', unsafe_allow_html=True)
        tcol1, tcol2, tcol3 = st.columns([0.44, 0.28, 0.28])
        with tcol1:
            # Pass the current scope as `index` so a view remount (e.g. returning
            # to Home) can't reset the selector while session state still points at
            # a specific document.
            scope_index = (
                scope_options.index(st.session_state.selected_scope)
                if st.session_state.selected_scope in scope_options else 0
            )
            st.session_state.selected_scope = st.selectbox(
                "Document Scope", scope_options, index=scope_index,
                label_visibility="collapsed"
            )
        with tcol2:
            st.session_state.deep_search = st.toggle(
                "Deep Search", value=st.session_state.deep_search,
                help="Broader retrieval and stronger reranking."
            )
        with tcol3:
            st.session_state.show_citations = st.toggle(
                "Citations", value=st.session_state.show_citations,
                help="Show document sources & page numbers with answers."
            )

        # ── Query Input Handling ─────────────────────────
        user_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

        chat_input_val = st.chat_input("Ask anything about your documents...")
        if chat_input_val:
            user_query = chat_input_val

        # Execute Dynamic RAG Query
        if user_query:
            if not st.session_state.messages:
                ensure_active_conversation()
                auto_title = generate_title_from_query(user_query)
                for c in st.session_state.conversations:
                    if c["id"] == st.session_state.active_conv_id:
                        c["title"] = auto_title
                        break

            st.session_state.messages.append({"role": "user", "content": user_query})

            primary_doc = st.session_state.indexed_files[0]["name"] if st.session_state.indexed_files else "Document"
            doc_stats = f"{len(st.session_state.all_chunks)} chunks in index • {relative_time(st.session_state.indexed_files[-1].get('upload_time', '')) if st.session_state.indexed_files else '—'}"

            scoped_docs = None if st.session_state.selected_scope == "All Documents" else [st.session_state.selected_scope]

            # Stream inside the chat scroll box (stream_slot is its last child) so
            # the live exchange appears above the docked scope bar and input
            # instead of after them in document flow.
            with stream_slot.container():
                st.markdown(f"""
                <div class="user-msg-container">
                    <div class="user-msg-bubble">{html.escape(user_query)}</div>
                    <div class="user-monogram-avatar">A</div>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("Analyzing document context & generating response..."):
                    stream_gen, retrieved_docs = stream_rag_answer(
                        question=user_query,
                        chat_history=st.session_state.messages,
                        session_id=st.session_state.session_id,
                        all_chunks=st.session_state.all_chunks,
                        selected_docs=scoped_docs,
                        deep_search=st.session_state.deep_search,
                        mode=st.session_state.active_mode,
                        model_name=st.session_state.selected_model,
                        top_k_dense=st.session_state.top_k_dense,
                        top_k_bm25=st.session_state.top_k_bm25
                    )

                    response_placeholder = st.empty()
                    full_response = ""

                    for chunk in stream_gen:
                        full_response += chunk
                        response_placeholder.markdown(full_response + " ▌")

                    response_placeholder.markdown(full_response)

                serialized_sources = []
                for doc, score in retrieved_docs:
                    serialized_sources.append({
                        "source": doc.metadata.get("source", "Document"),
                        # Default None, not 0: an absent page means the whole
                        # document, and 0 would render as "Page 1".
                        "page": doc.metadata.get("page"),
                        "content": doc.page_content[:220].strip().replace("\n", " "),
                        "score": float(score) if score is not None else None
                    })

                st.session_state.messages.append({
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": full_response,
                    "primary_doc": primary_doc,
                    "doc_stats": doc_stats,
                    "sources": serialized_sources
                })

                persist_active_messages()

                st.rerun()


# ══════════════════════════════════════════════════════════
# SHARED: DOCUMENTS TABLE (search, filter, ask, delete)
# ══════════════════════════════════════════════════════════
def render_documents_table():
    """Renders the searchable document table with working row actions."""
    uploads = st.file_uploader(
        "Upload documents", type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True, label_visibility="collapsed",
        help="Drag & Drop PDF, DOCX, TXT or MD files (Max 200MB)"
    )
    if st.session_state.failed_uploads:
        st.warning("Some uploads failed: " + ", ".join(sorted(st.session_state.failed_uploads)))
        if st.button("Retry failed uploads", key="retry_failed_table"):
            st.session_state.failed_uploads = set()
            st.rerun()
    if uploads:
        with st.spinner("Processing & indexing document(s)..."):
            if index_uploaded_files(uploads):
                st.rerun()

    lcol1, lcol2 = st.columns([0.7, 0.3])
    with lcol1:
        search_query = st.text_input("Search documents by title...", "")
    with lcol2:
        filter_type = st.selectbox("Filter Format", ["All Formats", "PDF", "DOCX", "TXT", "Markdown"])

    st.markdown("---")

    st.markdown("""
    <div style="display:grid; grid-template-columns: 2.5fr 1fr 1fr 1fr 1fr 2fr; font-size:12px; font-weight:700; color:#8C7A6B; text-transform:uppercase; padding:0 16px 8px 16px;">
        <span>Document Name</span>
        <span>Format</span>
        <span>Size</span>
        <span>Chunks</span>
        <span>Status</span>
        <span>Actions</span>
    </div>
    """, unsafe_allow_html=True)

    filtered_docs = list(st.session_state.indexed_files)
    if search_query:
        filtered_docs = [d for d in filtered_docs if search_query.lower() in d["name"].lower()]
    filtered_docs = filter_documents_by_format(filtered_docs, filter_type)

    if not filtered_docs:
        st.info("No documents found in your library.")
        return

    for doc in filtered_docs:
        row = st.columns([2.5, 1, 1, 1, 1, 2])
        with row[0]:
            st.markdown(f"<span style='display:flex; align-items:center; gap:6px;'>{get_icon('file-text', size=14, color='#7A695C')} <strong>{html.escape(doc['name'])}</strong></span>", unsafe_allow_html=True)
        with row[1]:
            st.markdown(
                f"<span class='indexed-pill-green' style='background:#F1E9DD; color:#4A3326;'>{doc.get('type', '—')}</span>",
                unsafe_allow_html=True
            )
        with row[2]:
            st.caption(doc.get("size_str", "—"))
        with row[3]:
            st.caption(str(doc.get("chunks_count", "—")))
        with row[4]:
            st.markdown(f"<span class='indexed-pill-green'>{doc.get('status', 'Indexed')}</span>", unsafe_allow_html=True)
        with row[5]:
            act1, act2 = st.columns([0.6, 0.4])
            with act1:
                if st.button("Ask Questions", key=f"ask_{doc['id']}", use_container_width=True,
                             help="Open a chat scoped to this document"):
                    st.session_state.selected_scope = doc["name"]
                    st.session_state.active_nav = "Home"
                    st.rerun()
            with act2:
                if st.button("Delete", key=f"del_{doc['id']}", use_container_width=True,
                             help="Remove this document and its chunks"):
                    remove_document(doc["name"])
                    st.rerun()


# ══════════════════════════════════════════════════════════
# VIEW 2: KNOWLEDGE BASE — manage documents & collections
# ══════════════════════════════════════════════════════════
if st.session_state.active_nav in ["Knowledge Base", "Documents", "Library"]:
    st.markdown('<div class="main-editorial-title">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-editorial-sub">Upload, search, scope, and manage the verified documents in your workspace.</div>', unsafe_allow_html=True)
    render_documents_table()

# ══════════════════════════════════════════════════════════
# VIEW 3: SETTINGS & PREFERENCES
# ══════════════════════════════════════════════════════════
elif st.session_state.active_nav == "Settings":
    st.markdown('<div class="main-editorial-title">Workspace Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-editorial-sub">Configure the AI engine, retrieval parameters, and knowledge base storage.</div>', unsafe_allow_html=True)

    st.markdown('<div class="studio-panel-box">', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:16px; font-weight:700; color:#2D2319; margin-bottom:12px; display:flex; align-items:center; gap:8px;'>{get_icon('cpu', size=18, color='#4A3326')} AI Model Engine</div>", unsafe_allow_html=True)
    # Keyless widgets assigned back to plain session keys: a widget-keyed value is
    # deleted when the Settings view unmounts, silently reverting user changes.
    st.session_state.selected_model = st.selectbox(
        "Active Groq Model", FALLBACK_MODELS,
        index=FALLBACK_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in FALLBACK_MODELS else 0,
        help="Used for chat answers, summaries, comparisons, and generated artifacts."
    )
    st.caption("Changes apply from the next message onward — there is nothing to save.")

    st.markdown(f"<div style='font-size:16px; font-weight:700; color:#2D2319; margin:20px 0 12px 0; display:flex; align-items:center; gap:8px;'>{get_icon('zap', size=18, color='#4A3326')} Hybrid Retrieval Parameters</div>", unsafe_allow_html=True)
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.session_state.top_k_dense = st.slider(
            "Dense Vector Candidates (Top-K)", min_value=10, max_value=60,
            value=st.session_state.top_k_dense,
            help="How many vector-search candidates are fused and reranked per query.")
    with rcol2:
        st.session_state.top_k_bm25 = st.slider(
            "BM25 Keyword Candidates (Top-K)", min_value=10, max_value=60,
            value=st.session_state.top_k_bm25,
            help="How many keyword-search candidates are fused and reranked per query.")

    st.markdown(f"<div style='font-size:16px; font-weight:700; color:#2D2319; margin:20px 0 12px 0; display:flex; align-items:center; gap:8px;'>{get_icon('message-square', size=18, color='#4A3326')} Answer Style</div>", unsafe_allow_html=True)
    st.session_state.active_mode = st.selectbox(
        "Response mode", ["smart", "strict"],
        index=0 if st.session_state.active_mode == "smart" else 1,
        format_func=lambda m: "Smart — may add general knowledge (flagged)" if m == "smart"
        else "Strict — only the uploaded documents",
        help="Strict refuses to answer anything that is not in the retrieved context."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="studio-panel-box">', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:16px; font-weight:700; color:#2D2319; margin-bottom:12px; display:flex; align-items:center; gap:8px;'>{get_icon('database', size=18, color='#4A3326')} Data & Storage Management</div>", unsafe_allow_html=True)
    st.caption(f"Session vector store: {st.session_state.session_id}")

    if st.button("Purge Session Vector Database & Clear Documents", use_container_width=True,
                 disabled=not (st.session_state.indexed_files or st.session_state.messages)):
        clear_session_storage(st.session_state.session_id)
        session_store.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.indexed_files = []
        st.session_state.all_chunks = []
        st.session_state.failed_uploads = set()
        st.session_state.selected_scope = "All Documents"
        st.session_state.conversations = []
        st.session_state.active_conv_id = None
        st.session_state.session_id = session_store.rotate_session_id()
        st.session_state._db_loaded = True
        st.success("Session vector store purged. Documents and chat cleared.")

    orphan_count = len(list_session_collections())
    st.caption(f"Vector collections on disk: {orphan_count}")
    if orphan_count > 1:
        st.warning(f"{orphan_count} vector collections exist on disk — leftover sessions are never reused.")
        if st.button("Delete ALL vector collections (every session)", type="primary"):
            deleted = purge_all_vector_stores()
            st.success(f"Deleted {deleted} collection(s).")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
