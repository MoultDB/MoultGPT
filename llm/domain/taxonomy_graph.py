# llm/domain/taxonomy_graph.py
"""
Taxonomy graph + name dictionary for Arthropoda.

- Loads a big CSV with hierarchical paths and synonyms.
- Builds a directed taxonomy graph (NetworkX).
- Builds a strict name index:
    * accepts only realistic arthropod taxon names
      (Genus, Genus species, and a small set of common names).
    * aggressively filters out generic junk like "Data", "Area", "China", "Thorax".
- Exposes a function to find arthropod taxa in free text.

Used by the gating layer to:
    - detect whether a paper mentions arthropods,
    - route or block questions about moulting.
"""

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import networkx as nx

# Path to taxonomy CSV
TAXON_CSV_PATH = os.getenv(
    "MOULTGPT_TAXON_CSV",
    "data/arthropod_taxonomy.csv",
)


def _normalize_name(name: str) -> str:
    """Lowercase, strip, collapse spaces, remove trailing punctuation."""
    name = name.strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[.,;:]+$", "", name)
    return name


@lru_cache(maxsize=1)
def load_taxonomy() -> pd.DataFrame:
    """Load the arthropod taxonomy CSV."""
    if not os.path.exists(TAXON_CSV_PATH):
        raise FileNotFoundError(f"Taxonomy CSV not found at {TAXON_CSV_PATH}")
    df = pd.read_csv(TAXON_CSV_PATH)

    required = [
        "id",
        "path",
        "ncbi_canonical_name",
        "inat_canonical_name",
        "gbif_canonical_name",
        "gbif_synonyms_names",
        "ncbi_synonyms_names",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in taxonomy CSV: {missing}")
    return df


@lru_cache(maxsize=1)
def build_taxonomy_graph() -> nx.DiGraph:
    """
    Create a directed taxonomy graph from 'path'.
    path examples: '1', '1.35', '1.35.2', etc.
    """
    df = load_taxonomy()
    G = nx.DiGraph()

    # Add nodes
    for _, row in df.iterrows():
        node_id = int(row["id"])
        G.add_node(
            node_id,
            path=str(row["path"]),
            ncbi_name=row.get("ncbi_canonical_name") or "",
            inat_name=row.get("inat_canonical_name") or "",
            gbif_name=row.get("gbif_canonical_name") or "",
        )

    # path -> id
    path_to_id: Dict[str, int] = {
        str(row["path"]): int(row["id"]) for _, row in df.iterrows()
    }

    # Add edges parent → child from path hierarchy
    for node_id, attrs in G.nodes(data=True):
        path = attrs["path"]
        if "." in path:
            parent_path = path.rsplit(".", 1)[0]
        else:
            parent_path = None

        if parent_path and parent_path in path_to_id:
            parent_id = path_to_id[parent_path]
            G.add_edge(parent_id, node_id)

    return G


def _is_valid_taxon_name(raw_name: str) -> bool:
    """
    Return True only if 'raw_name' looks like a realistic arthropod taxon.

    Aggressively filters:
    - generic single words ("Data", "Area", "China", "Region", ...)
    - anatomical / generic terms ("Thorax", "Abdomen", "Growth", ...)
    - things with digits or special symbols.

    Accepts:
    - Binomial scientific names: "Genus species".
    - Single Genus names: "Fuxianhuia" (capitalized, rest lowercase).
    - A small set of allowed common names: "spider", "crab", etc.
    """
    if not isinstance(raw_name, str):
        return False

    n = raw_name.strip()
    if len(n) < 4:
        return False

    # Reject if it contains digits or weird symbols (keep letters, spaces, hyphen)
    if re.search(r"[^A-Za-z\s\-]", n):
        return False

    n_low = n.lower()
    tokens = n_low.split()

    # Strict whitelist of common names that we want to accept as taxa
    allowed_common = {
        "spider", "spiders",
        "scorpion", "scorpions",
        "mite", "mites",
        "tick", "ticks",
        "ant", "ants",
        "bee", "bees",
        "wasp", "wasps",
        "crab", "crabs",
        "shrimp", "shrimps",
        "lobster", "lobsters",
        "centipede", "centipedes",
        "millipede", "millipedes",
    }
    if n_low in allowed_common:
        return True

    # Junk generic single words we never want as taxa
    banned_single = {
        "data", "area", "areas", "region", "regions",
        "china", "country", "state", "locality", "localities",
        "record", "records", "number", "sample", "samples",
        "information", "dataset", "table", "appendix",
    }
    if n_low in banned_single:
        return False

    # Generic anatomical / contextual words we don't want as taxa
    banned_anatomy = {
        "thorax", "abdomen", "cephalothorax", "head", "trunk",
        "segment", "segments", "segmented",
        "growth", "development", "juvenile", "larval",
        "fossil", "formation", "deposit", "layer",
    }
    if n_low in banned_anatomy:
        return False

    # Accept binomial species names: two tokens "genus species"
    if len(tokens) == 2:
        # both alphabetic tokens and not generic banned words
        if all(t.isalpha() for t in tokens):
            if tokens[0] not in banned_single and tokens[1] not in banned_single:
                return True

    # Accept single Genus names only if nicely capitalised in original string
    # e.g. "Fuxianhuia", "Isoxys"
    if len(tokens) == 1 and n[0].isupper() and n[1:].islower():
        return True

    # Everything else: reject
    return False


@lru_cache(maxsize=1)
def build_name_index() -> Dict[str, Dict[str, Any]]:
    """
    Build a strict name index:

        normalized_name -> {
            "taxon_id": int,
            "path": str,
            "source": "ncbi"|"inat"|"gbif"|...,
            "raw_name": original string from CSV
        }

    Uses canonical names + synonyms from all available columns.
    Applies _is_valid_taxon_name to avoid junk entries like "Data", "Area", etc.
    """
    df = load_taxonomy()
    name_index: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        taxon_id = int(row["id"])
        path = str(row["path"])

        raw_names: List[Tuple[str, str]] = []

        # Canonical names
        for col, src in [
            ("ncbi_canonical_name", "ncbi"),
            ("inat_canonical_name", "inat"),
            ("gbif_canonical_name", "gbif"),
        ]:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                raw_names.append((val, src))

        # GBIF synonyms
        syn_gbif = row.get("gbif_synonyms_names")
        if isinstance(syn_gbif, str) and syn_gbif.strip():
            for name in syn_gbif.split(";"):
                if name.strip():
                    raw_names.append((name, "gbif_synonym"))

        # NCBI synonyms
        syn_ncbi = row.get("ncbi_synonyms_names")
        if isinstance(syn_ncbi, str) and syn_ncbi.strip():
            for name in syn_ncbi.split(";"):
                if name.strip():
                    raw_names.append((name, "ncbi_synonym"))

        for raw_name, source in raw_names:
            # Filter aggressively before normalizing
            if not _is_valid_taxon_name(raw_name):
                continue

            norm = _normalize_name(raw_name)
            if len(norm) < 3:
                continue

            # Keep first occurrence for simplicity
            if norm not in name_index:
                name_index[norm] = {
                    "taxon_id": taxon_id,
                    "path": path,
                    "source": source,
                    "raw_name": raw_name.strip(),
                }

    return name_index


def _get_top_group(
    taxon_id: int,
    G: Optional[nx.DiGraph] = None,
    df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """
    Climb up the tree to the first child of root ('path' == '1', Arthropoda)
    and return its name (e.g. 'Chelicerata', 'Mandibulata', etc.).
    """
    if G is None:
        G = build_taxonomy_graph()
    if df is None:
        df = load_taxonomy()

    id_to_row = {int(r["id"]): r for _, r in df.iterrows()}

    current = taxon_id
    if current not in id_to_row:
        return None

    while True:
        row = id_to_row.get(current)
        if row is None:
            return None

        path = str(row["path"])

        # Case 1: actual root 1 = Arthropoda
        if path == "1":
            return (
                row.get("ncbi_canonical_name")
                or row.get("inat_canonical_name")
                or row.get("gbif_canonical_name")
                or "Arthropoda"
            )

        # Case 2: immediate child of root (e.g. '1.35')
        if "." in path and path.count(".") == 1:
            name = (
                row.get("ncbi_canonical_name")
                or row.get("inat_canonical_name")
                or row.get("gbif_canonical_name")
            )
            return name or None

        parents = list(G.predecessors(current))
        if not parents:
            return None
        current = parents[0]


def find_arthropod_taxa_in_text(text: str) -> List[Dict[str, Any]]:
    """
    Find arthropod taxa in arbitrary text.

    Matching logic (strict, token-based):
        - build unigrams and bigrams from normalized text.
        - match only if:
            * single-token name appears as unigram, or
            * two-token name appears as bigram.

    This avoids substring nonsense like "data" matching "database".
    """
    if not text:
        return []

    name_index = build_name_index()
    G = build_taxonomy_graph()
    df = load_taxonomy()

    text_norm = _normalize_name(text)
    tokens = text_norm.split()
    if not tokens:
        return []

    unigrams = set(tokens)
    bigrams = set(" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1))

    candidates: List[str] = []
    for norm_name in name_index.keys():
        if " " in norm_name:
            # binomial / multi-word name → require exact bigram match
            if norm_name in bigrams:
                candidates.append(norm_name)
        else:
            # single-token name → require exact unigram match
            if norm_name in unigrams:
                candidates.append(norm_name)

    matches: List[Dict[str, Any]] = []
    for norm_name in candidates:
        meta = name_index[norm_name]
        taxon_id = meta["taxon_id"]
        top_group = _get_top_group(taxon_id, G=G, df=df)
        matches.append(
            {
                "matched_name": meta["raw_name"],
                "norm_name": norm_name,
                "taxon_id": taxon_id,
                "path": meta["path"],
                "top_group": top_group,
                "source": meta["source"],
            }
        )

    # Deduplicate by taxon_id
    unique: Dict[int, Dict[str, Any]] = {}
    for m in matches:
        if m["taxon_id"] not in unique:
            unique[m["taxon_id"]] = m

    return list(unique.values())


if __name__ == "__main__":
    print("[TAXON] Loading taxonomy and building graph...")
    df = load_taxonomy()
    G = build_taxonomy_graph()
    name_idx = build_name_index()
    print(f"[TAXON] Rows: {len(df)} | Nodes: {G.number_of_nodes()} | Names indexed: {len(name_idx)}")

    test_q = "What moulting traits are reported for Hurdiidae and Kerygmachela in this paper?"
    found = find_arthropod_taxa_in_text(test_q)
    print(f"[TAXON] Query: {test_q}")
    print(f"[TAXON] Matches: {found}")
