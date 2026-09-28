"""
Candidate Generation / Blocking Module for Business Entity Resolution.
Builds high-recall, high-selectivity inverted index partitions per country.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from src.preprocessor import clean_name_tokens, clean_addr_tokens, extract_numbers, get_compressed_name


class CountryPartitionBlocker:
    """
    Manages multi-key inverted index blocking for a specific country partition (e.g. US, India, France).
    """

    def __init__(self, max_postings_per_key: int = 150, max_candidates_per_entity: int = 8):
        self.max_postings_per_key = max_postings_per_key
        self.max_candidates_per_entity = max_candidates_per_entity
        self.inverted_index = defaultdict(list)
        # Store metadata for fast feature lookup: idx -> (entity_id, name, addr)
        self.records: List[Tuple[str, str, str]] = []

    def get_blocking_keys(self, name: str, addr: str) -> List[Tuple[str, float]]:
        """
        Generate weighted blocking keys for a record.
        Returns list of (key, weight).
        """
        keys = []
        ntoks = clean_name_tokens(name)
        atoks = clean_addr_tokens(addr)
        nums = extract_numbers(addr)
        comp = get_compressed_name(name)

        # 1. Exact first two name tokens (very strong key)
        if len(ntoks) >= 2:
            keys.append((f"n12:{ntoks[0]}_{ntoks[1]}", 3.0))
            keys.append((f"n1:{ntoks[0]}", 1.0))
        elif len(ntoks) == 1:
            keys.append((f"n1:{ntoks[0]}", 1.5))

        # 2. Compressed name prefix
        if len(comp) >= 8:
            keys.append((f"nc8:{comp[:8]}", 3.0))
            keys.append((f"nc5:{comp[:5]}", 1.5))
        elif len(comp) >= 5:
            keys.append((f"nc5:{comp[:5]}", 2.0))

        # 3. Address number + first address token
        if nums and atoks:
            keys.append((f"na:{nums[0]}_{atoks[0]}", 3.0))
            if len(nums) >= 2:
                keys.append((f"num2:{nums[0]}_{nums[1]}", 2.5))

        # 4. Name first token + Address first token
        if ntoks and atoks:
            keys.append((f"na11:{ntoks[0]}_{atoks[0]}", 3.5))

        # 5. First two address tokens
        if len(atoks) >= 2:
            keys.append((f"a12:{atoks[0]}_{atoks[1]}", 2.0))

        return keys

    def add_record(self, entity_id: str, name: str, addr: str) -> int:
        """Add candidate record (from S2 or S3) to the index."""
        rec_idx = len(self.records)
        self.records.append((entity_id, name, addr))

        keys = self.get_blocking_keys(name, addr)
        for key, _ in keys:
            postings = self.inverted_index[key]
            # Prune if key becomes too frequent (e.g. generic name)
            if len(postings) < self.max_postings_per_key:
                postings.append(rec_idx)

        return rec_idx

    def query_candidates(
        self, name: str, addr: str, min_score: float = 1.5
    ) -> List[Tuple[str, str, str, float]]:
        """
        Query candidate records for a Source 1 entity.
        Returns list of (candidate_id, candidate_name, candidate_addr, score) sorted by score descending.
        Prunes weak matches below min_score to keep candidate sets compact and high-precision.
        """
        keys = self.get_blocking_keys(name, addr)
        cand_scores = defaultdict(float)

        for key, weight in keys:
            postings = self.inverted_index.get(key)
            if postings and len(postings) <= self.max_postings_per_key:
                # Add score contribution
                for rec_idx in postings:
                    cand_scores[rec_idx] += weight

        if not cand_scores:
            return []

        # Filter candidates meeting minimum accumulated key weight
        filtered = [item for item in cand_scores.items() if item[1] >= min_score]
        if not filtered:
            return []

        # Sort candidate indices by accumulated score descending
        sorted_indices = sorted(filtered, key=lambda x: x[1], reverse=True)[
            : self.max_candidates_per_entity
        ]

        results = []
        for rec_idx, score in sorted_indices:
            eid, cname, caddr = self.records[rec_idx]
            results.append((eid, cname, caddr, score))

        return results

    def clear(self):
        """Free memory after processing a country partition."""
        self.inverted_index.clear()
        self.records.clear()
