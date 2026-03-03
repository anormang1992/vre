# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
ConceptResolver — lemmatize + direct name-lookup resolver for VRE.

Concepts declared on the decorator are matched to canonical primitive names
via case-insensitive exact match and spaCy lemmatization. No embeddings or
similarity scoring.
"""

from __future__ import annotations

from functools import lru_cache

from vre.core.graph import PrimitiveRepository


@lru_cache(maxsize=1)
def _nlp():
    """
    Return the cached spaCy language model, loading it on first call.
    """
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )


def lemmatize(text: str) -> list[str]:
    """
    Lemmatize text, removing stopwords, punctuation, and short tokens.

    Returns a list of lowercased lemmas suitable for passing to VRE.resolve().
    """
    doc = _nlp()(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 1
    ]


class ConceptResolver:
    """
    Aligns concept names to canonical VRE primitive names via lemmatization
    and direct name lookup. No embeddings or similarity scoring.

    For each input concept:
    1. Try direct case-insensitive match against known primitive names.
    2. Lemmatize and try each resulting lemma against the same name map.
    3. Return None if no match found — the caller decides how to handle unknowns.
    """

    def __init__(self, repository: PrimitiveRepository) -> None:
        """
        Initialize the resolver with a primitive repository.
        """
        self._repo = repository

    def build_name_map(self) -> dict[str, str]:
        """
        Map lowercased primitive names to their canonical form.
        """
        return {name.lower(): name for name in self._repo.list_names()}

    @staticmethod
    def lookup(concept: str, name_map: dict[str, str]) -> str | None:
        """
        Return canonical name for concept, or None if not found.
        """
        # Try direct lowercase match first
        if concept.lower() in name_map:
            return name_map[concept.lower()]
        # Try each lemma
        for lemma in lemmatize(concept):
            if lemma in name_map:
                return name_map[lemma]
        return None

    def resolve(self, concepts: list[str]) -> list[str]:
        """
        Return canonical primitive names for all matching concepts.
        """
        name_map = self.build_name_map()
        result, seen = [], set()
        for concept in concepts:
            canonical = self.lookup(concept, name_map)
            if canonical and canonical not in seen:
                seen.add(canonical)
                result.append(canonical)
        return result
