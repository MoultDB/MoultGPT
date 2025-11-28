# llm/debug_preprocess.py

import argparse
import sys
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parent
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from pipeline.processor import input_to_text  # type: ignore
from pipeline.gating import (  # type: ignore
    analyze_paper_for_moulting,
    route_query_for_paper,
)

# Hard-coded test queries for the gating logic
TEST_QUERIES = [
    # 1. Explicit arthropod taxa
    "What moulting traits are reported for Hurdiidae and Kerygmachela in this paper?",
    # 2. Generic moulting question (should pass only if paper has arthropods)
    "Extract all information related to moulting of the species in this paper.",
    # 3. Vertebrate moulting (should be rejected)
    "How often do birds moult their feathers?",
    # 4. Generic non-moulting question (should be rejected)
    "Which species is described in this paper?",
    # 5. Common-name alias (spider)
    "Summarise all moulting traits of the spider described in this paper.",
    # 6. Totally off-topic
    "What is the GDP of France?",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Debug pipeline: DOI/PDF -> text -> paper analysis (taxa + summary) "
            "-> query gating."
        )
    )
    parser.add_argument("--doi", type=str, default=None, help="DOI of the paper")
    parser.add_argument("--pdf", type=str, default=None, help="Path to local PDF")
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Email for Unpaywall (overrides default in downloader.py)",
    )
    args = parser.parse_args()

    if not args.doi and not args.pdf:
        print("ERROR: Provide either --doi or --pdf")
        sys.exit(1)

    # ── STEP 1: full text extraction ─────────────────────────────
    print("\n[STEP 1] Extracting full text...")
    full_text = input_to_text(doi=args.doi, pdf_path=args.pdf, email=args.email)

    if not full_text or len(full_text.strip()) < 100:
        print("[ERROR] Extracted text is empty or too short.")
        sys.exit(1)

    print(f"[INFO] Extracted text length: {len(full_text)} characters")
    print("\n[PREVIEW] First 1000 characters:\n")
    print(full_text[:1000])
    print("\n" + "-" * 80 + "\n")

    # ── STEP 2: paper-level analysis (taxa + summary) ────────────
    print("[STEP 2] analyze_paper_for_moulting(...)")
    analysis = analyze_paper_for_moulting(full_text)

    paper_taxa = analysis["paper_taxa"]
    summary = analysis["summary"]
    n_summary = analysis["n_summary"]
    paper_has_arthropods = analysis["paper_has_arthropods"]
    paper_is_relevant = analysis["paper_is_relevant"]

    print(f"[INFO] Arthropod taxa in paper: {len(paper_taxa)}")
    if paper_taxa:
        names = [m["matched_name"] for m in paper_taxa]
        print(f"[INFO]  Example taxa: {names[:10]}")
    print(f"[INFO] Relevant sentences in summary: {n_summary}")
    print(
        f"[INFO] paper_has_arthropods={paper_has_arthropods} | "
        f"paper_is_relevant={paper_is_relevant}"
    )

    print("\n[SUMMARY OUTPUT]\n")
    print(summary)
    print("\n" + "-" * 80 + "\n")

    # ── STEP 3: query-level gating against this paper ────────────
    print("[STEP 3] Testing queries against routing (route_query_for_paper)...\n")

    for q in TEST_QUERIES:
        print("-" * 80)
        print(f"QUERY: {q}")
        decision = route_query_for_paper(q, paper_taxa, n_summary)
        print(
            f"  allow={decision['allow']} | "
            f"stage={decision['stage']} | "
            f"reason={decision['reason']}"
        )
        print(f"  message: {decision['message']}")

    print("\n" + "-" * 80)
    print("[DONE] Debug pipeline completed.")


if __name__ == "__main__":
    main()
