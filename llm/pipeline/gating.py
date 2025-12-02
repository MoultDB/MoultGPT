# llm/pipeline/gating.py
"""
Gating layer for MoultGPT (LLM side).

Responsibilities:
- Inspect a full-text paper:
    * detect arthropod taxa via taxonomy graph,
    * extract moulting-related sentences,
    * decide if the paper is relevant enough.
- Inspect a user query and the paper context to decide:
    * whether the question is in scope (arthropod moulting only),
    * whether the paper is suitable for generic moulting questions.

Exposed functions:
    - analyze_paper_for_moulting(full_text: str) -> dict
    - route_query_for_paper(user_prompt: str, paper_taxa: list, n_relevant_sentences: int) -> dict
"""

from typing import Any, Dict, List

from domain.taxonomy_graph import find_arthropod_taxa_in_text  # type: ignore
from pipeline.summarization import extract_relevant_sentences  # type: ignore


# ------------------------ Paper-level analysis ------------------------


def analyze_paper_for_moulting(
    full_text: str,
    snippet_chars: int = 10000,
    min_relevant_sentences: int = 10,
) -> Dict[str, Any]:
    """
    Analyze a paper and return:

        {
            "summary": str (moulting-related sentences),
            "paper_taxa": list of taxa dicts,
            "n_summary": int,
            "paper_has_arthropods": bool,
            "paper_is_relevant": bool,
        }

    Logic:
        - Taxa are detected on a snippet of the text (title+abstract+start).
        - Relevant sentences are extracted on the full text.
        - paper_is_relevant = n_summary >= min_relevant_sentences
    """
    if not full_text or len(full_text.strip()) < 50:
        return {
            "summary": "",
            "paper_taxa": [],
            "n_summary": 0,
            "paper_has_arthropods": False,
            "paper_is_relevant": False,
        }

    snippet = full_text[:snippet_chars]
    paper_taxa = find_arthropod_taxa_in_text(snippet)

    summary = extract_relevant_sentences(full_text)
    if summary:
        lines = [l for l in summary.splitlines() if l.strip()]
    else:
        lines = []
    n_summary = len(lines)

    paper_has_arthropods = len(paper_taxa) > 0
    paper_is_relevant = n_summary >= min_relevant_sentences

    return {
        "summary": summary,
        "paper_taxa": paper_taxa,
        "n_summary": n_summary,
        "paper_has_arthropods": paper_has_arthropods,
        "paper_is_relevant": paper_is_relevant,
    }


# ------------------------ Question-level helpers ------------------------


def _mentions_moulting(question: str) -> bool:
    q = question.lower()
    moulting_terms = [
        "moult",
        "moulting",
        "molting",
        "molt",
        "ecdysis",
        "exuvia",
        "exuviae",
        "exuvium",
        "instar",
        "stadium",
        "cuticle",
        "exoskeleton",
        "larva",
        "larval",
        "nymph",
        "juvenile",
    ]
    return any(t in q for t in moulting_terms)


def _has_arthropod_terms(question: str) -> bool:
    q = question.lower()
    arthropod_terms = [
        "arthropod",
        "arthropods",
        "insect",
        "insects",
        "spider",
        "spiders",
        "scorpion",
        "scorpions",
        "crustacean",
        "crustaceans",
        "myriapod",
        "myriapods",
        "chelicerate",
        "chelicerates",
        "hexapod",
        "hexapods",
        "trilobite",
        "trilobites",
        "panarthropod",
        "panarthropods",
        # generic "species" etc. lo lasciamo fuori, troppo generico
    ]
    return any(t in q for t in arthropod_terms)


def _has_vertebrate_moulting(question: str) -> bool:
    q = question.lower()

    vertebrate_terms = [
        "bird",
        "birds",
        "mammal",
        "mammals",
        "human",
        "humans",
        "reptile",
        "reptiles",
        "snake",
        "snakes",
        "lizard",
        "lizards",
        "frog",
        "frogs",
        "amphibian",
        "amphibians",
        "fish",
        "fishes",
    ]
    moulting_terms = [
        "moult",
        "moulting",
        "molting",
        "molt",
        "shed",
        "shedding",
        "feather",
        "feathers",
        "skin",
        "fur",
        "hair",
        "plumage",
        "coat",
    ]

    return any(v in q for v in vertebrate_terms) and any(m in q for m in moulting_terms)


# ------------------------ Query + paper gating ------------------------


def route_query_for_paper(
    user_prompt: str,
    paper_taxa: List[Dict[str, Any]],
    n_relevant_sentences: int,
    min_relevant_sentences: int = 3,
) -> Dict[str, Any]:
    """
    Decide whether a query should be routed to the LLM given:
        - the user prompt,
        - the taxa detected in the paper,
        - the number of relevant moulting sentences.

    Returns a dict:
        {
            "allow": bool,
            "stage": "question_gate" | "paper_gate" | "ok",
            "reason": str,
            "message": str,
        }
    """

    q = (user_prompt or "").strip()
    if not q:
        return {
            "allow": False,
            "stage": "question_gate",
            "reason": "empty_query",
            "message": "Empty query; please provide a question.",
        }

    has_moulting = _mentions_moulting(q)
    has_arthropod = _has_arthropod_terms(q)
    has_vertebrate = _has_vertebrate_moulting(q)

    # 1) Obvious out-of-scope: vertebrate moulting
    if has_vertebrate:
        return {
            "allow": False,
            "stage": "question_gate",
            "reason": "vertebrate_moulting_out_of_scope",
            "message": (
                "The question appears to concern moulting in vertebrates (birds, mammals, reptiles, etc.). "
                "MoultGPT is restricted to moulting in arthropods. Please specify an arthropod taxon "
                "or rephrase your question in terms of arthropod moulting."
            ),
        }

    # 2) Must have at least some connection to moulting OR arthropods
    if not has_moulting and not has_arthropod:
        return {
            "allow": False,
            "stage": "question_gate",
            "reason": "no_moulting_or_arthropods",
            "message": (
                "The question does not appear to concern arthropods or moulting. "
                "MoultGPT is specialized in moulting traits of arthropods. "
                "Please mention moulting and/or an arthropod group."
            ),
        }

    # 3) Paper-level gate: is the paper actually moulting-related?
    if n_relevant_sentences < min_relevant_sentences:
        return {
            "allow": False,
            "stage": "paper_gate",
            "reason": "paper_not_moulting_related",
            "message": (
                "The provided article does not seem to contain enough moulting-related content "
                "to answer questions reliably (too few relevant sentences were detected)."
            ),
        }

    # 4) Archetype A: explicit arthropod-moulting question
    if has_moulting and has_arthropod:
        return {
            "allow": True,
            "stage": "ok",
            "reason": "arthropod_taxa_in_query",
            "message": "Query and paper pass all gates; request can be sent to the LLM.",
        }

    # 5) Archetype B: generic moulting question with no explicit arthropod in the wording
    #    → allowed only if the paper itself has arthropod taxa.
    if has_moulting and not has_arthropod:
        if paper_taxa:
            return {
                "allow": True,
                "stage": "ok",
                "reason": "generic_moulting_on_arthropod_paper",
                "message": "Query and paper pass all gates; request can be sent to the LLM.",
            }
        else:
            return {
                "allow": False,
                "stage": "paper_gate",
                "reason": "no_arthropod_taxa_in_paper",
                "message": (
                    "The question concerns moulting, but the article does not appear to discuss arthropod taxa. "
                    "MoultGPT is restricted to arthropod moulting. Please provide an article about arthropods "
                    "or specify an arthropod group."
                ),
            }

    # Fallback safety
    return {
        "allow": False,
        "stage": "question_gate",
        "reason": "unclassified_out_of_scope",
        "message": (
            "The question could not be confidently classified as an arthropod-moulting query. "
            "Please rephrase and explicitly mention moulting in arthropods."
        ),
    }


if __name__ == "__main__":
    # Minimal smoke test
    txt = "Trilobites undergo ecdysis and shed their exuviae multiple times during growth."
    analysis = analyze_paper_for_moulting(txt)
    print("[GATING TEST] analysis:", analysis)

    q = "Extract all moulting traits of the trilobite described in this paper."
    decision = route_query_for_paper(q, analysis["paper_taxa"], analysis["n_summary"])
    print("[GATING TEST] decision:", decision)
