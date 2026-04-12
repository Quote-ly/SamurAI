# SamurAI

SamurAI is the VirtualDojo team's AI-powered assistant, embedded directly in Microsoft Teams. Its purpose is to be a helpful, autonomous member of the team -- handling DevOps troubleshooting, GitHub workflow management, CRM queries, FedRAMP compliance, social media, and proactive follow-ups so the team can focus on building the product.

## What SamurAI does

SamurAI is not just a chatbot. It is an autonomous agent that can investigate issues end-to-end, take action on behalf of the team, and follow up without being prompted.

### Troubleshooting and infrastructure
- Query Google Cloud logs, metrics, and Cloud Run service status across all environments
- View GCP billing/cost breakdown by service (read-only, via BigQuery billing export)
- Correlate errors with deployments by tracking revision names and timestamps
- Distinguish real regressions from draining/shutdown noise after deploys
- Sync and read source code from GitHub repos to trace bugs back to the code
- Cross-reference logs, code, and service status to deliver root cause analysis

### GitHub workflow
- Review PRs, list issues, check recent commits, view commit diffs across all Quote-ly repos
- Create GitHub issues (always checking for duplicates first)
- Manage GitHub Projects V2 (create items, update Status/Priority fields)
- Suggest the `autofix` label on quotely-data-service bugs when appropriate (with user approval)
- Close duplicate or erroneous issues (with a reason)

### CRM and business data
- Query VirtualDojo CRM data (contacts, accounts, opportunities, quotes, compliance records)
- Handle OAuth flow for user authentication to the CRM
- Read-only by default; creating/updating/deleting records requires human approval

### Communication and team coordination
- Send 1:1 Teams messages to team members
- Create scheduled/recurring background tasks that run autonomously
- Follow up on sent messages (e.g., check if someone reviewed a PR after being reminded)
- Escalate when things haven't been addressed after multiple attempts

### FedRAMP compliance
- Collect automated evidence from GCP (IAM, Cloud Run configs, KMS, audit logs, SCC findings)
- Generate and update OSCAL packages (SSP, POA&M)
- Review code against FedRAMP control families (SC-7, SC-12, CM-6, SC-18, AC-8)
- Track remediation SLAs and flag overdue items

### Social media
- Draft, preview, schedule, and publish posts to LinkedIn, X/Twitter, and other platforms via Ayrshare
- Generate images with VirtualDojo brand colors
- Enforces preview-before-publish flow; only Cyrus and Devin can approve posts

### File handling
- Process uploaded spreadsheets (Excel/CSV) -- fill columns, edit specific cells, return modified files
- Always verifies changes with read-back before reporting success

## Autonomy rules

SamurAI acts independently on read-only operations, communications, and scheduling. It requires explicit human approval (Devin or Cyrus) before:
- Modifying production infrastructure or deploying services
- Creating/closing/merging GitHub PRs
- Modifying CRM records
- Publishing social media posts
- Deleting persistent data

## Tech stack

- **Runtime**: Python 3.12, aiohttp, Microsoft Bot Framework SDK
- **AI**: LangGraph agent with Google Gemini (`gemini-3.1-pro-preview`), LangChain tools
- **Scheduling**: APScheduler (AsyncIOScheduler) for background tasks
- **Persistence**: SQLite on GCS FUSE mount (`/data`) for tasks, conversation refs, team roster
- **Memory**: LangMem three-tier memory (core/team/user) with background extraction
- **Hosting**: Google Cloud Run (project: `virtualdojo-samurai`, region: `us-central1`)

## Key architecture

- `app.py` -- Bot entrypoint, message routing, Teams integration, error handling
- `agent.py` -- LangGraph agent graph, system prompt, tool binding, `run_agent()` entry point
- `scheduler.py` -- APScheduler background task execution, conversation ref resolution, retry logic
- `task_store.py` -- SQLite persistence for tasks, conversation refs, team roster
- `tools/` -- All agent tools:
  - `gcp_logging.py`, `gcp_cloudrun.py`, `gcp_monitoring.py` -- GCP infrastructure + billing
  - `github.py` -- GitHub issues, PRs, commits, commit diffs, projects
  - `repo_sync.py` -- Sync and read source code from GitHub repos
  - `background_tasks.py` -- Create/manage scheduled tasks
  - `teams_messaging.py` -- Send 1:1 Teams messages
  - `virtualdojo_mcp.py` -- VirtualDojo CRM integration
  - `social_media.py` -- Ayrshare social media publishing
  - `fedramp.py`, `fedramp_oscal.py`, `fedramp_docs.py` -- FedRAMP compliance
  - `file_handler.py` -- Spreadsheet processing
  - `google_search.py` -- Web search
  - `database.py` -- Database tools
- `cards/` -- Adaptive Card builders and action handlers (social media previews, etc.)
- `memory.py` -- LangGraph checkpointing and LangMem three-tier memory store

## GitHub repos SamurAI can access

- `Quote-ly/quotely-data-service` -- Main data service (FastAPI + Vue.js CRM)
- `Quote-ly/virtualdojo_cli` -- VirtualDojo CLI tool
- `Quote-ly/SamurAI` -- This bot
- `Quote-ly/Fedramp` -- FedRAMP compliance documentation and OSCAL packages

## Autofix label (quotely-data-service)

The `autofix` label on quotely-data-service issues triggers an automated Claude-based TDD bug fix attempt (via the `claude_automation/bugfix/` workflow in that repo). SamurAI may suggest applying the label but must never apply it without explicit user approval.

Good candidates for autofix:
- Backend data/logic bugs with a clear error trace (NOT NULL violations, type mismatches, missing defaults, query filter bugs, wrong field references)
- API endpoint bugs where the error and expected behavior are unambiguous
- Regex/pattern matching fixes (error sanitization, input parsing)
- Missing or incorrect DB column defaults, constraints, or migrations
- Off-by-one errors, wrong status codes, missing null checks
- Test gaps where the fix is adding coverage for an existing behavior

Bad candidates for autofix:
- Frontend/UI bugs (Vue components, CSS, layout) -- requires visual verification
- Multi-tenant authorization or access control changes -- too security-sensitive
- Alembic migrations on production data -- need manual review and rollback planning
- Business logic changes that require product/UX decisions
- Performance issues -- profiling needed, not just code changes
- Anything touching payment, compliance, or PII handling

## Three-tier memory system

SamurAI learns from every interaction through three memory tiers:

| Tier | Namespace | Who reads it | What's stored |
|------|-----------|-------------|---------------|
| **Core** | `("core",)` | Everyone (including future external users) | Successful tool patterns, troubleshooting recipes, error resolutions |
| **Team** | `("team", "virtualdojo")` | VirtualDojo team only | Project decisions, infrastructure facts, internal processes |
| **User** | `("memories", "{user_id}")` | That user only | Personal preferences, communication style, role context |

After each conversation, three background extractors run automatically to populate each tier. The bot also has explicit memory tools (`manage_core_memory`, `manage_team_memory`, `manage_memory`) to save knowledge during conversations. All tiers are searched and injected into the system prompt on every message.

Tool calls and their outcomes are logged and included in the extraction payload, so the core extractor can learn successful multi-tool patterns over time.

## GCP projects

- `virtualdojo-samurai` -- This bot's infrastructure
- `virtualdojo-fedramp-dev` -- FedRAMP dev environment
- `virtualdojo-fedramp-prod` -- FedRAMP production environment

## Deployment

Deploy to Cloud Run using source-based deployment (builds via Dockerfile):

```bash
gcloud run deploy samurai-bot --source . --region=us-central1 --project=virtualdojo-samurai
```

This preserves all existing config (env vars, secrets, volume mounts, scaling). Only the container image is updated.

If auth has expired: `gcloud auth login`

### Cloud Run configuration

- Min instances: 1 (always warm), Max instances: 20
- Memory: 2Gi, CPU: 1, CPU throttling: disabled
- Persistent storage: GCS FUSE bucket `samurai-bot-data` mounted at `/data`
- Execution environment: gen2, startup CPU boost enabled

## Running tests

```bash
python -m pytest tests/ -v
```

## Known operational notes

- APScheduler runs in-process; jobs are in-memory and rebuilt from SQLite on restart
- Recursion limit is 50 for both interactive and background tasks
- One-shot tasks get 1 automatic retry on failure (60s delay)
- Conversation refs are resolved through `bg_task_` parent chains for sub-task delivery
- Background tasks are tagged with `is_background_task=True` so the agent executes directly without conversational back-and-forth
- Tool calls are logged to stdout (`[agent] tool_calls` and `[agent] tool_result`) for observability in Cloud Logging
- If the bot hits the recursion limit, it asks the user what to focus on instead of failing silently
- SQLite over GCS FUSE shows occasional `OutOfOrderError` on journal files -- this is expected
- GCP billing export is configured on `virtualdojo-samurai.billing_export` (table populates daily, env var: `GCP_BILLING_TABLE`)
