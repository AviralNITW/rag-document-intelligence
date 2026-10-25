from typing import List, Any, Generator, Tuple
import logging

from services.rag_service import get_llm_instance
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


MAX_SUMMARY_CHUNKS = 24
MAX_COMPARISON_CHUNKS = 14
MAX_ARTIFACT_CHUNKS = 20


def select_representative_chunks(chunks: List[Any], max_count: int = MAX_SUMMARY_CHUNKS) -> List[Any]:
    """Selects a representative sample of chunks preserving introduction, body progression, and conclusion.

    Unlike naive subsampling which drops 90%+ of document content, this preserves:
    - 100% of chunks for small-to-medium documents (<= max_count)
    - Full introductory context, evenly spaced topical progression, and complete concluding sections for large files.
    """
    if not chunks:
        return []
    if len(chunks) <= max_count:
        return list(chunks)

    # Reserve head (intro/overview) and tail (conclusions/summary)
    head_size = min(3, len(chunks))
    tail_size = min(3, len(chunks) - head_size)

    head = chunks[:head_size]
    tail = chunks[-tail_size:] if tail_size > 0 else []
    middle_pool = chunks[head_size:-tail_size] if tail_size > 0 else chunks[head_size:]

    needed_middle = max_count - len(head) - len(tail)
    if needed_middle > 0 and middle_pool:
        step = max(1, len(middle_pool) // needed_middle)
        sampled_middle = middle_pool[::step][:needed_middle]
    else:
        sampled_middle = []

    return head + sampled_middle + tail


def generate_document_summary(
    all_chunks: List[Any],
    doc_name: str,
    model_name: str = "qwen/qwen3.8-27b"
) -> Tuple[Generator[str, None, None], List[Any]]:
    """Generates an executive, structured summary of a specific document."""
    relevant_chunks = [c for c in all_chunks if c.metadata.get("source") == doc_name]
    if not relevant_chunks:
        relevant_chunks = all_chunks

    sample = select_representative_chunks(relevant_chunks, max_count=MAX_SUMMARY_CHUNKS)
    context_sample = "\n\n".join([c.page_content for c in sample])

    prompt = (
        f"You are RAGI Document Intelligence. Summarize the document '{doc_name}' with high precision.\n"
        "Format your output in this exact structure:\n\n"
        "### Summary\n[1-2 paragraph executive summary describing the core purpose and context]\n\n"
        "### 📑 Key Points\n"
        "- **Key Area 1:** Description\n"
        "- **Key Area 2:** Description\n"
        "- **Key Area 3:** Description\n\n"
        "### 💡 Takeaways & Next Steps\n[Closing concise paragraph]\n\n"
        f"Document Content:\n{context_sample}"
    )

    llm = get_llm_instance(model_name)

    def stream_gen():
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    return stream_gen(), sample[:5]


def generate_document_comparison(
    doc_a_name: str,
    doc_b_name: str,
    all_chunks: List[Any],
    model_name: str = "qwen/qwen3.8-27b"
) -> Generator[str, None, None]:
    """Generates a structured side-by-side comparison between two documents."""
    def sample_for(name: str) -> List[str]:
        chunks = [c for c in all_chunks if c.metadata.get("source") == name]
        sampled = select_representative_chunks(chunks, max_count=MAX_COMPARISON_CHUNKS)
        return [c.page_content for c in sampled]

    text_a = "\n\n".join(sample_for(doc_a_name)) or "No text found for Document A."
    text_b = "\n\n".join(sample_for(doc_b_name)) or "No text found for Document B."

    prompt = (
        f"You are RAGI Document Intelligence. Perform a rigorous comparative analysis between:\n"
        f"Document A: '{doc_a_name}'\nDocument B: '{doc_b_name}'\n\n"
        "Provide a structured analysis with the following headings:\n"
        "### 1. Comparative Overview\n"
        "### 2. Core Similarities\n"
        "### 3. Key Differences & Contrasts\n"
        "### 4. Distinct Advantages & Scope\n"
        "### 5. Final Synthesis\n\n"
        f"--- Document A Context ---\n{text_a}\n\n"
        f"--- Document B Context ---\n{text_b}"
    )

    llm = get_llm_instance(model_name)
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def generate_specialized_artifact(
    artifact_type: str,
    doc_name: str,
    all_chunks: List[Any],
    model_name: str = "qwen/qwen3.8-27b"
) -> Generator[str, None, None]:
    """Generates notes, technical reports, FAQs, or flashcards from a document."""
    scoped = [c for c in all_chunks if c.metadata.get("source") == doc_name]
    target_pool = scoped if scoped else all_chunks
    sampled_chunks = select_representative_chunks(target_pool, max_count=MAX_ARTIFACT_CHUNKS)
    context_text = "\n\n".join([c.page_content for c in sampled_chunks])

    instructions = {
        "Study Notes": "Create comprehensive, structured study notes with key definitions, formulas, and bulleted takeaways.",
        "Executive Summary": "Create a high-impact C-suite executive briefing with problem statement, findings, and strategic recommendations.",
        "Technical Report": "Create an in-depth technical analysis breaking down architecture, methodologies, technologies, and specifications.",
        "FAQ": "Create a comprehensive list of 8-10 Frequently Asked Questions (with detailed answers) based on this document.",
        "Flashcards": "Create 6-8 Question & Answer flashcard pairs formatted cleanly for rapid revision and active recall."
    }

    selected_instruction = instructions.get(artifact_type, "Create a structured analysis report.")

    prompt = (
        f"You are RAGI Document Intelligence. Your task is to generate: **{artifact_type}** for the document '{doc_name}'.\n"
        f"Instruction: {selected_instruction}\n\n"
        f"Document Content:\n{context_text}"
    )

    llm = get_llm_instance(model_name)
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content