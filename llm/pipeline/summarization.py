# llm/pipeline/summarization.py
"""
Sentence-level filtering + summarization for arthropod moulting content.

- Split full text into sentences.
- Keep only sentences that clearly talk about moulting / ecdysis / exuviae
  in an arthropod context.
- Cluster with TF-IDF + KMeans to pick representative sentences.

Design goal:
    Prefer false negatives (drop borderline stuff)
    over false positives (e.g. economics "growth / development" papers).
"""

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def _simple_sentence_split(text: str) -> List[str]:
    """Very simple sentence splitter: dot/question/exclamation + capital letter."""
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.?!])\s+(?=[A-Z])", text)


def extract_relevant_sentences(full_text: str, num_sentences: int = 20) -> str:
    """
    Extract biologically relevant sentences about arthropod moulting.

    Rules:
      - Keep only sentences that contain:
          * clear moulting-related lexicon, OR
          * eco/physio keywords co-occurring with arthropod context.
      - Explicitly ignore generic growth/development/survival-only sentences
        to avoid pulling in unrelated fields (economics, demography, etc.).
    """

    # Core moulting lexicon
    core_keywords = [
        "moult", "moulting", "molting", "molt",
        "ecdysis", "ecdysial", "exuvia", "exuviae", "exuvium", "exuvial",
        "instar", "stadium",
        "larva", "larval", "nymph", "juvenile",
        "cuticle", "exoskeleton", "sclerotization", "sclerotisation",
        "premoult", "pre-moult", "premolt", "pre-molt",
        "postmoult", "post-moult", "postmolt", "post-molt",
        "intermoult", "inter-moult", "intermolt", "inter-molt",
        "ecdysone", "ecdysteroid",
    ]

    # Auxiliary eco/physio terms (only valid if arthropod context also present)
    aux_keywords = [
        "desiccation",
        "aestivation",
        "water loss",
        "calcification",
    ]

    # Arthropod / invertebrate context cues
    arthropod_context = [
        "arthropod", "arthropods",
        "insect", "insects",
        "crustacean", "crustaceans",
        "spider", "spiders",
        "scorpion", "scorpions",
        "myriapod", "myriapods",
        "chelicerate", "chelicerates",
        "hexapod", "hexapods",
        "trilobite", "trilobites",
        "panarthropod", "panarthropods",
        "fuxianhuia", "isoxys", "leanchoilia",  # some frequent Cambrian culprits
        "larva", "larvae", "nymph", "juvenile",
    ]

    def is_relevant(sent: str) -> bool:
        s = sent.strip()
        if len(s) < 30:
            return False
        if len(s.split()) < 5:
            return False
        if "®" in s or "fig." in s.lower():
            return False

        s_low = s.lower()

        has_core = any(k in s_low for k in core_keywords)
        has_aux = any(k in s_low for k in aux_keywords)
        has_ctx = any(k in s_low for k in arthropod_context)

        # Only core moulting lexicon counts directly.
        if has_core:
            return True

        # Auxiliary terms alone are not enough; require arthropod context
        if has_aux and has_ctx:
            return True

        return False

    all_sentences = [s.strip() for s in _simple_sentence_split(full_text)]
    filtered = [s for s in all_sentences if is_relevant(s)]

    if not filtered:
        return ""

    # TF-IDF + KMeans clustering over filtered sentences
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(filtered)
    k = min(num_sentences, len(filtered))

    # Explicit n_init to silence sklearn FutureWarning
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

    summary: List[tuple[int, str]] = []
    for i in range(k):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if not cluster_indices.size:
            continue
        center = kmeans.cluster_centers_[i]
        # choose the sentence closest to cluster center (most representative)
        scores = X[cluster_indices] @ center.T
        closest_idx = cluster_indices[np.argmax(scores)]
        summary.append((closest_idx, filtered[closest_idx]))

    # Preserve original order of appearance
    summary.sort(key=lambda x: x[0])
    return "\n".join(s for _, s in summary)
