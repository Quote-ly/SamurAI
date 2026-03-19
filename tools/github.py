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


@tool
def github_list_issues(repo: str, state: str = "open", count: int = 10) -> str:
    """List issues for a GitHub repository.

    Args:
        repo: Repository in 'owner/repo' format.
        state: Issue state — 'open', 'closed', or 'all'.
        count: Number of issues to return (default 10).
    """
    issues = _github().get_repo(repo).get_issues(state=state, sort="updated")
    # Filter out pull requests (GitHub API returns PRs as issues too)
    results = []
    for issue in issues:
        if issue.pull_request is not None:
            continue
        labels = ", ".join(l.name for l in issue.labels) if issue.labels else "none"
        results.append(
            f"#{issue.number} {issue.title} ({issue.state}) "
            f"by {issue.user.login} | labels: {labels}"
        )
        if len(results) >= count:
            break

    return "\n".join(results) if results else f"No {state} issues found in {repo}."


@tool
def github_get_issue_details(repo: str, issue_number: int) -> str:
    """Get details of a specific GitHub issue including comments.

    Args:
        repo: Repository in 'owner/repo' format.
        issue_number: The issue number.
    """
    issue = _github().get_repo(repo).get_issue(issue_number)
    labels = ", ".join(l.name for l in issue.labels) if issue.labels else "none"
    assignees = ", ".join(a.login for a in issue.assignees) if issue.assignees else "unassigned"

    result = (
        f"Title: {issue.title}\n"
        f"State: {issue.state}\n"
        f"Author: {issue.user.login}\n"
        f"Labels: {labels}\n"
        f"Assignees: {assignees}\n"
        f"Created: {issue.created_at}\n"
        f"Body:\n{issue.body or '(empty)'}"
    )

    comments = list(issue.get_comments()[:5])
    if comments:
        result += "\n\nRecent comments:"
        for c in comments:
            result += f"\n- {c.user.login}: {c.body[:200]}"

    return result


@tool
def github_create_issue(
    repo: str, title: str, body: str, labels: str = ""
) -> str:
    """Create a new GitHub issue.

    Args:
        repo: Repository in 'owner/repo' format.
        title: The issue title.
        body: The issue body/description (supports markdown).
        labels: Comma-separated label names to apply (optional).
    """
    repo_obj = _github().get_repo(repo)
    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else []
    issue = repo_obj.create_issue(title=title, body=body, labels=label_list)
    return f"Created issue #{issue.number}: {issue.title}\nURL: {issue.html_url}"


@tool
def github_list_workflow_runs(
    repo: str, status: str = "", count: int = 10
) -> str:
    """List recent GitHub Actions workflow runs for a repository.

    Args:
        repo: Repository in 'owner/repo' format.
        status: Filter by status — 'completed', 'in_progress', 'queued', 'failure', 'success', or '' for all.
        count: Number of runs to return (default 10).
    """
    repo_obj = _github().get_repo(repo)
    kwargs = {}
    if status:
        kwargs["status"] = status
    runs = repo_obj.get_workflow_runs(**kwargs)[:count]

    if not runs:
        return f"No workflow runs found in {repo}."

    lines = []
    for run in runs:
        duration = ""
        if run.created_at and run.updated_at:
            delta = run.updated_at - run.created_at
            duration = f" ({delta.total_seconds():.0f}s)"
        lines.append(
            f"{run.name} | {run.conclusion or run.status} | "
            f"run_id={run.id} #{run.run_number} on {run.head_branch}{duration} | "
            f"{run.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
    return "\n".join(lines)


@tool
def github_get_workflow_run_details(repo: str, run_id: int) -> str:
    """Get details of a specific GitHub Actions workflow run, including failed job info.

    Args:
        repo: Repository in 'owner/repo' format.
        run_id: The workflow run ID (a large number like 14358032881, NOT the run number). Use the run_id from github_list_workflow_runs output.
    """
    repo_obj = _github().get_repo(repo)
    run = repo_obj.get_workflow_run(run_id)

    result = (
        f"Workflow: {run.name}\n"
        f"Status: {run.status} | Conclusion: {run.conclusion}\n"
        f"Branch: {run.head_branch}\n"
        f"Triggered by: {run.event} ({run.triggering_actor.login if run.triggering_actor else 'unknown'})\n"
        f"Run #: {run.run_number}\n"
        f"URL: {run.html_url}\n"
    )

    jobs = list(run.jobs())
    if jobs:
        result += "\nJobs:"
        for job in jobs:
            result += f"\n  {job.name}: {job.conclusion or job.status}"
            if job.conclusion == "failure":
                # Show failed steps
                for step in job.steps:
                    if step.conclusion == "failure":
                        result += f"\n    FAILED step: {step.name}"

    return result
