"""
Unit tests for vre.core.shell — parse_bash_primitives.
"""

from vre.builtins.shell import SHELL_ALIASES, parse_bash_primitives


def test_rm_maps_to_delete_file():
    assert parse_bash_primitives("rm -rf /tmp/foo") == ["delete", "file"]


def test_mkdir_maps_to_create_directory():
    assert parse_bash_primitives("mkdir /tmp/newdir") == ["create", "directory"]


def test_cat_maps_to_read_file():
    assert parse_bash_primitives("cat /etc/hosts") == ["read", "file"]


def test_curl_maps_to_network_request():
    assert parse_bash_primitives("curl https://example.com") == ["network", "request"]


def test_chmod_maps_to_permission_file():
    assert parse_bash_primitives("chmod 755 script.sh") == ["permission", "file"]


def test_unknown_command_returns_empty():
    assert parse_bash_primitives("unknowncmd arg1 arg2") == []


def test_empty_string_returns_empty():
    assert parse_bash_primitives("") == []


def test_path_prefix_stripped():
    """Full path to executable is handled correctly."""
    assert parse_bash_primitives("/usr/bin/rm -f file.txt") == ["delete", "file"]


def test_all_aliases_are_lists_of_strings():
    for cmd, concepts in SHELL_ALIASES.items():
        assert isinstance(concepts, list), f"{cmd!r} value is not a list"
        assert all(isinstance(c, str) for c in concepts), f"{cmd!r} has non-string concepts"


def test_ls_maps_to_list_directory():
    assert parse_bash_primitives("ls -la") == ["list", "directory"]
