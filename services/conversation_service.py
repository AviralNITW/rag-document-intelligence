import uuid
from datetime import datetime
from typing import List, Dict, Any


def create_new_conversation(initial_title: str = "New Chat") -> Dict[str, Any]:
    """Creates a new conversation object."""
    return {
        "id": str(uuid.uuid4()),
        "title": initial_title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [],
    }


def generate_title_from_query(query: str) -> str:
    """Generates a clean title from the user's first query."""
    clean = query.strip()
    words = clean.split()
    if len(words) <= 5:
        return clean.capitalize()

    lower = clean.lower()
    if any(w in lower for w in ["summarize", "summary"]):
        return "Document Summary & Insights"
    elif any(w in lower for w in ["compare", "difference"]):
        return "Comparative Document Analysis"
    elif any(w in lower for w in ["skill", "experience", "resume"]):
        return "Resume & Experience Breakdown"
    elif any(w in lower for w in ["architecture", "design"]):
        return "System Architecture Review"
    else:
        return " ".join(words[:4]).capitalize() + "..."