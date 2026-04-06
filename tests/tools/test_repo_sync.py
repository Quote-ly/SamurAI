"""Tests for tools/repo_sync.py — local repo sync and code reading tools."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest


# --- sync_repo ---


def test_sync_repo_rejects_unlisted_repo():
    from tools.repo_sync import sync_repo

    result = sync_repo.invoke({"repo": "Evil/hacker-repo", "branch": "main"})
    assert "not a whitelisted repo" in result


@patch("tools.github._github_token", return_value="fake-token")
@patch("tools.repo_sync._get_remote_sha", return_value=None)
def test_sync_repo_handles_unreachable_branch(mock_sha, mock_token):
    from tools.repo_sync import sync_repo

    result = sync_repo.invoke(
        {"repo": "Quote-ly/quotely-data-service", "branch": "nonexistent"}
    )
    assert "Could not reach" in result


@patch("tools.github._github_token", return_value="fake-token")
@patch("tools.repo_sync._get_local_sha", return_value="abc12345")
@patch("tools.repo_sync._get_remote_sha", return_value="abc12345")
def test_sync_repo_skips_when_up_to_date(mock_remote, mock_local, mock_token):
    from tools.repo_sync import sync_repo

    result = sync_repo.invoke(
        {"repo": "Quote-ly/quotely-data-service", "branch": "main"}
    )
    assert "Already up to date" in result
    assert "abc12345" in result


@patch("subprocess.run")
@patch("tools.repo_sync._get_local_sha", return_value=None)
@patch("tools.repo_sync._get_remote_sha", return_value="def67890")
def test_sync_repo_clones_when_missing(mock_remote, mock_local, mock_run):
    from tools.repo_sync import sync_repo

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with patch("tools.github._github_token", return_value="fake-token"):
        result = sync_repo.invoke(
            {"repo": "Quote-ly/quotely-data-service", "branch": "main"}
        )

    assert "Synced successfully" in result
    assert "def67890" in result
    # Verify git clone was called with --depth 1
    clone_call = [c for c in mock_run.call_args_list if "clone" in str(c)]
    assert len(clone_call) > 0


@patch("subprocess.run")
@patch("tools.repo_sync._get_local_sha", return_value="old111")
@patch("tools.repo_sync._get_remote_sha", return_value="new222")
def test_sync_repo_reclones_when_stale(mock_remote, mock_local, mock_run):
    from tools.repo_sync import sync_repo

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with patch("tools.github._github_token", return_value="fake-token"):
        result = sync_repo.invoke(
            {"repo": "Quote-ly/quotely-data-service", "branch": "development"}
        )

    assert "Synced successfully" in result


@patch("subprocess.run")
@patch("tools.repo_sync._get_local_sha", return_value=None)
@patch("tools.repo_sync._get_remote_sha", return_value="abc123")
def test_sync_repo_handles_clone_failure(mock_remote, mock_local, mock_run):
    from tools.repo_sync import sync_repo

    mock_run.return_value = MagicMock(
        returncode=128, stdout="", stderr="fatal: repository not found"
    )

    with patch("tools.github._github_token", return_value="fake-token"):
        result = sync_repo.invoke(
            {"repo": "Quote-ly/quotely-data-service", "branch": "main"}
        )

    assert "Error cloning" in result


# --- read_repo_file ---


def test_read_repo_file_rejects_unlisted_repo():
    from tools.repo_sync import read_repo_file

    result = read_repo_file.invoke(
        {"file_path": "main.py", "repo": "Evil/repo", "branch": "main"}
    )
    assert "not a whitelisted repo" in result


def test_read_repo_file_not_synced():
    from tools.repo_sync import read_repo_file

    result = read_repo_file.invoke(
        {"file_path": "main.py", "repo": "Quote-ly/quotely-data-service", "branch": "nonexistent-branch-xyz"}
    )
    assert "not synced yet" in result.lower() or "Call sync_repo" in result


def test_read_repo_file_reads_content(tmp_path):
    from tools.repo_sync import read_repo_file, _repo_dir

    repo = "Quote-ly/quotely-data-service"
    branch = "main"
    local_dir = _repo_dir(repo, branch)

    # Create a fake repo with a file
    os.makedirs(local_dir, exist_ok=True)
    test_file = os.path.join(local_dir, "test.py")
    with open(test_file, "w") as f:
        f.write("print('hello world')")

    try:
        result = read_repo_file.invoke(
            {"file_path": "test.py", "repo": repo, "branch": branch}
        )
        assert "hello world" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_read_repo_file_not_found(tmp_path):
    from tools.repo_sync import read_repo_file, _repo_dir

    repo = "Quote-ly/quotely-data-service"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    try:
        result = read_repo_file.invoke(
            {"file_path": "nonexistent.py", "repo": repo, "branch": branch}
        )
        assert "File not found" in result
    finally:
        os.rmdir(local_dir)


def test_read_repo_file_truncates_large_files(tmp_path):
    from tools.repo_sync import read_repo_file, _repo_dir

    repo = "Quote-ly/quotely-data-service"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "big.txt")
    with open(test_file, "w") as f:
        f.write("x" * 60000)

    try:
        result = read_repo_file.invoke(
            {"file_path": "big.txt", "repo": repo, "branch": branch}
        )
        assert "truncated" in result
        assert len(result) < 55000
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


# --- search_repo_code ---


def test_search_repo_code_rejects_unlisted_repo():
    from tools.repo_sync import search_repo_code

    result = search_repo_code.invoke(
        {"query": "test", "repo": "Evil/repo", "branch": "main"}
    )
    assert "not a whitelisted repo" in result


def test_search_repo_code_not_synced():
    from tools.repo_sync import search_repo_code

    result = search_repo_code.invoke(
        {"query": "test", "repo": "Quote-ly/quotely-data-service", "branch": "nonexistent-xyz"}
    )
    assert "not synced yet" in result.lower() or "Call sync_repo" in result


def test_search_repo_code_finds_matches(tmp_path):
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "Quote-ly/quotely-data-service"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "app.py")
    with open(test_file, "w") as f:
        f.write("allow_origins=['*']\nother_line\n")

    try:
        result = search_repo_code.invoke(
            {"query": "allow_origins", "repo": repo, "branch": branch}
        )
        assert "allow_origins" in result
        assert "app.py" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


def test_search_repo_code_no_matches(tmp_path):
    from tools.repo_sync import search_repo_code, _repo_dir

    repo = "Quote-ly/quotely-data-service"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    test_file = os.path.join(local_dir, "empty.py")
    with open(test_file, "w") as f:
        f.write("# nothing here\n")

    try:
        result = search_repo_code.invoke(
            {"query": "ZZZZNOTFOUND", "repo": repo, "branch": branch}
        )
        assert "No matches found" in result
    finally:
        os.remove(test_file)
        os.rmdir(local_dir)


# --- list_repo_files ---


def test_list_repo_files_rejects_unlisted_repo():
    from tools.repo_sync import list_repo_files

    result = list_repo_files.invoke(
        {"path": "", "repo": "Evil/repo", "branch": "main"}
    )
    assert "not a whitelisted repo" in result


def test_list_repo_files_not_synced():
    from tools.repo_sync import list_repo_files

    result = list_repo_files.invoke(
        {"path": "", "repo": "Quote-ly/quotely-data-service", "branch": "nonexistent-xyz"}
    )
    assert "not synced yet" in result.lower() or "Call sync_repo" in result


def test_list_repo_files_shows_contents(tmp_path):
    from tools.repo_sync import list_repo_files, _repo_dir

    repo = "Quote-ly/quotely-data-service"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(os.path.join(local_dir, "app"), exist_ok=True)

    with open(os.path.join(local_dir, "main.py"), "w") as f:
        f.write("# entry point")
    with open(os.path.join(local_dir, "requirements.txt"), "w") as f:
        f.write("flask\n")

    try:
        result = list_repo_files.invoke(
            {"path": "", "repo": repo, "branch": branch}
        )
        assert "main.py" in result
        assert "requirements.txt" in result
        assert "app/" in result
    finally:
        import shutil
        shutil.rmtree(local_dir)


def test_list_repo_files_hides_dotfiles(tmp_path):
    from tools.repo_sync import list_repo_files, _repo_dir

    repo = "Quote-ly/quotely-data-service"
    branch = "main"
    local_dir = _repo_dir(repo, branch)
    os.makedirs(local_dir, exist_ok=True)

    with open(os.path.join(local_dir, ".git"), "w") as f:
        f.write("")
    with open(os.path.join(local_dir, "visible.py"), "w") as f:
        f.write("")

    try:
        result = list_repo_files.invoke(
            {"path": "", "repo": repo, "branch": branch}
        )
        assert ".git" not in result
        assert "visible.py" in result
    finally:
        import shutil
        shutil.rmtree(local_dir)
