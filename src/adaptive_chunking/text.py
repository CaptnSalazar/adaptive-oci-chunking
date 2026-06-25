from __future__ import annotations

import math
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
REFERENCE_RE = re.compile(
    r"\b(?:section|sec\.|figure|fig\.|table|appendix|equation|eq\.)\s+\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,:;&()/-]{3,80})$", re.MULTILINE)
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def normalize_space(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def token_count(text: str) -> int:
    return len(words(text))


def sentences(text: str) -> list[str]:
    return [
        normalize_space(match.group(0))
        for match in SENTENCE_RE.finditer(text)
        if match.group(0).strip()
    ]


def cosine_bow(left: str, right: str) -> float:
    left_counts = Counter(words(left))
    right_counts = Counter(words(right))
    if not left_counts or not right_counts:
        return 0.0
    keys = left_counts.keys() & right_counts.keys()
    dot = sum(left_counts[key] * right_counts[key] for key in keys)
    left_norm = sum(value * value for value in left_counts.values()) ** 0.5
    right_norm = sum(value * value for value in right_counts.values()) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def lexical_entropy(text: str) -> float:
    terms = words(text)
    if not terms:
        return 0.0
    counts = Counter(terms)
    total = len(terms)
    entropy = -sum((count / total) * math.log(count / total, 2) for count in counts.values())
    max_entropy = math.log(total, 2) if total > 1 else 1.0
    return entropy / max_entropy if max_entropy else 0.0


def find_references(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in REFERENCE_RE.finditer(text)]


def find_block_boundaries(text: str) -> set[int]:
    boundaries = {0, len(text)}
    for match in re.finditer(r"\n\s*\n", text):
        boundaries.add(match.end())
    for match in HEADING_RE.finditer(text):
        boundaries.add(match.start())
    return boundaries
