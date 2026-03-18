"""Tools for interacting with GitHub repositories via GitHub App auth."""

import os
import time

from langchain_core.tools import tool


def _github():
    """Authenticate as a GitHub App installation and return a Github client."""
    import jwt
    from github import Github, GithubIntegration

    app_id = os.environ["GITHUB_APP_ID"]
    private_key = os.environ["GITHUB_APP_PRIVATE_KEY"]

    integration = GithubIntegration(app_id, private_key)

    # Get the installation for the Quote-ly org
    installations = integration.get_installations()
    if not installations:
        raise RuntimeError("GitHub App is not installed on any organization.")

    installation_id = installations[0].id
    access_token = integration.get_access_token(installation_id).token
    return Github(access_token)


@tool
def github_list_prs(repo: str, state: str = "open") -> str:
    """List pull requests for a GitHub repository.

    Args:
        repo: Repository in 'owner/repo' format.
        state: PR state — 'open', 'closed', or 'all'.
    """
    pulls = _github().get_repo(repo).get_pulls(state=state, sort="updated")[:10]

    if not pulls:
        return f"No {state} PRs found in {repo}."

    lines = []
    for p in pulls:
        lines.append(f"#{p.number} {p.title} ({p.state}) by {p.user.login}")
    return "\n".join(lines)


@tool
def github_get_pr_details(repo: str, pr_number: int) -> str:
    """Get details of a specific pull request including changed files.

    Args:
        repo: Repository in 'owner/repo' format.
        pr_number: The PR number.
    """
    pr = _github().get_repo(repo).get_pull(pr_number)
    files = [f.filename for f in pr.get_files()]
    return (
        f"Title: {pr.title}\n"
        f"Author: {pr.user.login}\n"
        f"State: {pr.state}\n"
        f"Branch: {pr.head.ref} -> {pr.base.ref}\n"
        f"Files changed ({len(files)}): {', '.join(files)}"
    )


@tool
def github_list_recent_commits(
    repo: str, branch: str = "main", count: int = 10
) -> str:
    """List recent commits on a branch.

    Args:
        repo: Repository in 'owner/repo' format.
        branch: Branch name (default 'main').
        count: Number of commits to return (default 10).
    """
    commits = _github().get_repo(repo).get_commits(sha=branch)[:count]

    lines = []
    for c in commits:
        short_sha = c.sha[:7]
        msg = c.commit.message.splitlines()[0]
        author = c.commit.author.name
        lines.append(f"{short_sha} {msg} — {author}")
    return "\n".join(lines)
