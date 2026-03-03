"""
Unit tests for ConceptResolver — lemmatize + direct name-lookup.
"""

from unittest.mock import MagicMock

from vre.core.grounding import ConceptResolver, lemmatize


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stub_repo(names: list[str]):
    """Return a stub repository that returns given names from list_names()."""
    repo = MagicMock()
    repo.list_names.return_value = names
    return repo


# ── lemmatize ────────────────────────────────────────────────────────────────

def test_lemmatize_plural_to_singular():
    result = lemmatize("files")
    assert "file" in result


def test_lemmatize_removes_stopwords():
    result = lemmatize("the files in the system")
    assert "the" not in result
    assert "file" in result


def test_lemmatize_lowercases():
    result = lemmatize("WRITING")
    assert all(t == t.lower() for t in result)


def test_lemmatize_removes_punctuation():
    result = lemmatize("read, write!")
    for token in result:
        assert token.isalpha(), f"Expected alpha token, got: {token!r}"


def test_lemmatize_returns_list_of_strings():
    result = lemmatize("create a directory")
    assert isinstance(result, list)
    assert all(isinstance(t, str) for t in result)


# ── ConceptResolver ───────────────────────────────────────────────────────────

def test_resolver_direct_match():
    """Exact lowercase name → returns that canonical name."""
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    assert resolver.resolve(["file"]) == ["file"]


def test_resolver_case_insensitive_match():
    """Uppercase input matches lowercase canonical name."""
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    assert resolver.resolve(["FILE"]) == ["file"]


def test_resolver_canonical_case_preserved():
    """When the repo returns mixed-case names, the canonical case is returned."""
    repo = _stub_repo(["File"])
    resolver = ConceptResolver(repo)
    # "File" stored → canonical is "File", looked up via "file" key
    result = resolver.resolve(["file"])
    assert result == ["File"]


def test_resolver_lemmatized_match():
    """'files' lemmatizes to 'file' → matches canonical 'file'."""
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    result = resolver.resolve(["files"])
    assert result == ["file"]


def test_resolver_unknown_returns_empty():
    """Concept not in graph → not returned."""
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    assert resolver.resolve(["unknownxyz123"]) == []


def test_resolver_deduplicates():
    """Same canonical name matched by different inputs → returned once."""
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    result = resolver.resolve(["file", "files"])
    assert result == ["file"]


def test_resolver_empty_concepts():
    """Empty input → empty output."""
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    assert resolver.resolve([]) == []


def test_resolver_empty_graph():
    """Empty graph → nothing resolves."""
    repo = _stub_repo([])
    resolver = ConceptResolver(repo)
    assert resolver.resolve(["file"]) == []


def test_resolver_multiple_concepts():
    """Multiple known concepts → all returned in order."""
    repo = _stub_repo(["file", "write"])
    resolver = ConceptResolver(repo)
    result = resolver.resolve(["file", "write"])
    assert "file" in result
    assert "write" in result


def test_resolver_lookup_returns_none_for_unknown():
    """
    lookup returns None for unrecognized concept.
    """
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    name_map = resolver._build_name_map()
    assert resolver.lookup("unknownxyz123", name_map) is None


def test_resolver_lookup_returns_canonical():
    """
    lookup returns the canonical name for a known concept.
    """
    repo = _stub_repo(["file"])
    resolver = ConceptResolver(repo)
    name_map = resolver._build_name_map()
    assert resolver.lookup("file", name_map) == "file"


def test_resolver_build_name_map_lowercases_keys():
    """_build_name_map keys are always lowercase."""
    repo = _stub_repo(["File", "Write"])
    resolver = ConceptResolver(repo)
    name_map = resolver._build_name_map()
    assert "file" in name_map
    assert "write" in name_map


