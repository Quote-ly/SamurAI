"""LangGraph agent wired to Gemini with GCP, GitHub, VirtualDojo CRM, and memory tools."""

import logging
import os
import time

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from memory import (
    get_checkpointer,
    get_memory_store,
    create_memory_tools,
    retrieve_relevant_memories,
    get_background_extractor,
    persist_memories,
)
from tools.gcp_logging import query_cloud_logs
from tools.gcp_monitoring import check_gcp_metrics
from tools.gcp_cloudrun import list_cloud_run_services
from tools.github import (
    github_list_prs,
    github_get_pr_details,
    github_list_recent_commits,
    github_list_issues,
    github_get_issue_details,
    github_create_issue,
    github_list_workflow_runs,
    github_get_workflow_run_details,
    github_close_issue,
    PROJECT_TOOLS,
)
from tools.virtualdojo_mcp import create_virtualdojo_tool, create_virtualdojo_list_tools
from tools.social_media import SOCIAL_TOOLS
from tools.google_search import google_search
from tools.background_tasks import BACKGROUND_TASK_TOOLS
from tools.teams_messaging import TEAMS_MESSAGING_TOOLS
from tools.fedramp import FEDRAMP_TOOLS
from tools.fedramp_docs import FEDRAMP_DOC_TOOLS
from tools.fedramp_oscal import FEDRAMP_OSCAL_TOOLS
from tools.repo_sync import REPO_SYNC_TOOLS

logger = logging.getLogger(__name__)

# ── Tool Groups ────────────────────────────────────────────────────────
# Core tools are always loaded. Other groups load dynamically based on the request.

TOOL_GROUPS = {
    "core": {
        "tools": [
            query_cloud_logs,
            list_cloud_run_services,
            check_gcp_metrics,
            google_search,
        ] + BACKGROUND_TASK_TOOLS,
        "keywords": [],  # Always loaded
    },
    "github": {
        "tools": [
            github_list_prs,
            github_get_pr_details,
            github_list_recent_commits,
            github_list_issues,
            github_get_issue_details,
            github_create_issue,
            github_list_workflow_runs,
            github_get_workflow_run_details,
            github_close_issue,
        ] + PROJECT_TOOLS,
        "keywords": [
            "github", "pr", "pull request", "issue", "commit", "ci/cd",
            "workflow", "actions", "deploy", "branch", "merge", "repo",
            "project board", "project items",
        ],
    },
    "fedramp": {
        "tools": FEDRAMP_TOOLS,
        "keywords": [
            "fedramp", "compliance", "evidence", "audit log review",
            "scc", "iam compliance", "log retention", "encryption",
            "vulnerability", "dependabot", "poam", "poa&m",
            "nist", "control family", "800-53",
        ],
    },
    "oscal": {
        "tools": FEDRAMP_OSCAL_TOOLS,
        "keywords": [
            "oscal", "ssp", "poam", "assessment result", "generate ssp",
            "migrate", "catalog lookup", "look up control", "render pdf",
            "validate package", "update control", "link evidence",
        ],
    },
    "fedramp_docs": {
        "tools": FEDRAMP_DOC_TOOLS,
        "keywords": [
            "fedramp document", "read document", "propose edit",
            "commit document", "review code", "fedramp doc",
            "search document", "list document",
        ],
    },
    "social": {
        "tools": SOCIAL_TOOLS,
        "keywords": [
            "social", "post", "linkedin", "twitter", "facebook",
            "instagram", "publish", "schedule post", "preview post",
            "ayrshare", "draft",
        ],
    },
    "teams": {
        "tools": TEAMS_MESSAGING_TOOLS,
        "keywords": [
            "send message", "send a message", "teams message",
            "message to", "team roster", "lookup member", "team member",
        ],
    },
    "repo": {
        "tools": REPO_SYNC_TOOLS,
        "keywords": [
            "sync repo", "sync the", "pull the code", "read code",
            "search code", "source code", "troubleshoot", "debug",
            "codebase", "main.py", "config.py", "list files",
        ],
    },
}

# Flat list of ALL tools (for ToolNode which needs to execute any tool)
ALL_TOOLS = []
_seen = set()
for group in TOOL_GROUPS.values():
    for t in group["tools"]:
        if id(t) not in _seen:
            _seen.add(id(t))
            ALL_TOOLS.append(t)

# Keep STATIC_TOOLS for backward compat in tests
STATIC_TOOLS = ALL_TOOLS


def _select_tool_groups(message: str) -> list:
    """Select which tool groups to activate based on the user's message."""
    msg_lower = message.lower()
    selected = list(TOOL_GROUPS["core"]["tools"])  # Always include core

    for name, group in TOOL_GROUPS.items():
        if name == "core":
            continue
        if any(kw in msg_lower for kw in group["keywords"]):
            selected.extend(group["tools"])

    # If no specific group matched, include github (most common)
    non_core_matched = len(selected) > len(TOOL_GROUPS["core"]["tools"])
    if not non_core_matched:
        selected.extend(TOOL_GROUPS["github"]["tools"])

    return selected

SYSTEM_PROMPT = (
    "You are SamurAI, a DevOps and CRM assistant in Microsoft Teams. "
    "You help the team check Google Cloud infrastructure, read logs, "
    "monitor services, review GitHub activity, and query VirtualDojo CRM data. "
    "Be concise and use markdown formatting when it helps readability.\n\n"
    "IMPORTANT — GCP project IDs you have access to:\n"
    "- virtualdojo-samurai (this bot)\n"
    "- virtualdojo-fedramp-dev (FedRAMP dev environment)\n"
    "- virtualdojo-fedramp-prod (FedRAMP production environment)\n"
    "When the user mentions 'fedramp dev' or 'dev', use project_id='virtualdojo-fedramp-dev'. "
    "When they mention 'fedramp prod' or 'prod', use project_id='virtualdojo-fedramp-prod'. "
    "When the user asks about Cloud Run services, logs, or metrics without specifying a project, "
    "default to BOTH fedramp-dev and fedramp-prod. The team does not care about the samurai bot's own services. "
    "Never query virtualdojo-samurai for Cloud Run services unless the user explicitly asks about the bot itself.\n"
    "Always use the exact project IDs above — never guess or construct project IDs.\n\n"
    "GitHub organization: Quote-ly\n"
    "IMPORTANT — You may ONLY access these GitHub repositories:\n"
    "- Quote-ly/quotely-data-service (main data service)\n"
    "- Quote-ly/virtualdojo_cli (VirtualDojo CLI tool)\n"
    "- Quote-ly/SamurAI (this bot's repo)\n"
    "- Quote-ly/Fedramp (FedRAMP compliance documentation and OSCAL packages)\n"
    "NEVER attempt to access any other repository. If the user asks about a repo not in this list, "
    "tell them it's not configured and list the repos you can access.\n"
    "When the user says 'data service' or 'quotely', use Quote-ly/quotely-data-service. "
    "When they say 'CLI' or 'vdojo cli', use Quote-ly/virtualdojo_cli. "
    "When they say just a repo name without 'Quote-ly/', prefix it with 'Quote-ly/'.\n"
    "IMPORTANT: Before creating a GitHub issue, ALWAYS search existing issues first using "
    "github_list_issues to check for duplicates or similar issues. Do NOT create redundant issues.\n"
    "You can close issues with github_close_issue, but ONLY for cleaning up duplicates or "
    "issues created in error. Always include a reason when closing.\n\n"
    "VirtualDojo CRM:\n"
    "You can query CRM data (contacts, accounts, opportunities, quotes, compliance records) "
    "using the virtualdojo_crm tool. Use virtualdojo_list_tools to discover available operations. "
    "Common tool_name values: 'search_records', 'list_objects', 'describe_object', "
    "'create_record', 'update_record', 'get_record'. "
    "If the user asks about CRM data and is not signed in, tell them to say 'connect to VirtualDojo' to authenticate. "
    "NEVER generate or fabricate a login URL yourself. The bot will automatically provide the correct sign-in link "
    "when the user says 'connect to VirtualDojo'.\n\n"
    "Deployment & Revision Intelligence:\n"
    "When analyzing Cloud Run logs after a deployment, always note the resource.labels.revision_name "
    "in the log filter to distinguish which revision errors come from. "
    "Errors on an OLD revision within 5-10 minutes of a deployment are likely draining/shutdown noise — "
    "not regressions. Common draining patterns include: 'RuntimeError: Event loop is closed', "
    "'Connection reset by peer', and SIGTERM-related errors. "
    "Only treat errors as regressions if they occur on the NEW (latest) revision AND after it became healthy. "
    "When reporting errors, always state which revision they came from so the user can tell old vs new apart. "
    "If the user asks about a deployment, check the service status first to identify the current revision, "
    "then filter logs by that revision.\n\n"
    "Each message includes the user's name and timezone in brackets at the start. "
    "Use their timezone when displaying times — convert UTC timestamps to their local time. "
    "For example, if the user is in America/New_York, show times in ET.\n\n"
    "Social Media (LinkedIn, X/Twitter, and more):\n"
    "You can draft, preview, schedule, and publish social media posts via Ayrshare.\n"
    "Available platforms: linkedin, twitter, facebook, instagram, tiktok, bluesky, "
    "threads, pinterest, reddit, youtube, telegram, snapchat, gmb.\n"
    "You can also generate images for posts using AI image generation.\n\n"
    "CRITICAL SOCIAL MEDIA RULES:\n"
    "1. ALWAYS call social_preview_post first to show the user a preview before posting.\n"
    "2. NEVER call social_publish_post or social_schedule_post unless the user explicitly confirms "
    "with words like 'approve', 'post it', 'looks good', 'yes', 'confirmed', or 'send it'.\n"
    "3. If the user wants changes, create a new preview with the edits.\n"
    "4. Only Cyrus and Devin are authorized to use social media tools.\n"
    "5. When generating images, incorporate VirtualDojo brand colors (terra cotta #B84A3C, "
    "black #1A1A1A) and clean, modern visual style.\n"
    "6. Social media tools require conversation_id and user_email parameters — "
    "pass these from the context provided in each message.\n\n"
    "IMPORTANT: When calling social media tools that accept a conversation_id parameter, "
    "ALWAYS pass the conversation_id from the context brackets at the start of the message.\n\n"
    "VirtualDojo Brand Voice (for drafting social media posts):\n"
    "- Tone: 'Strategic Rowdiness' — conversational, authoritative, candid GovCon insider\n"
    "- Lead with real scenarios and pain points, not feature lists\n"
    "- Short punchy paragraphs, rhetorical questions OK\n"
    "- Back claims with specifics (contract vehicles, percentages, real numbers)\n"
    "- Hashtags: #GovCon #GovernmentContracting #CMMC #FedRAMP #SEWP #NIST\n"
    "- X handle: @Virtualdojo_gov\n"
    "- NEVER say 'FedRAMP authorized' — say 'pursuing FedRAMP Moderate authorization'\n"
    "- NEVER say '100%' accuracy — say '99.9%+'\n"
    "- NEVER use generic SaaS speak or forced enthusiasm\n"
    "- NEVER use em dashes (—) in social media posts. Use periods, commas, or line breaks instead.\n\n"
    "Long-term Memory:\n"
    "You have a persistent memory system powered by LangMem.\n"
    "- Use manage_memory to save, update, or delete facts and preferences.\n"
    "- Use search_memory to find relevant past context.\n"
    "- Memories are also automatically extracted from conversations in the background.\n"
    "- Save memories proactively when users share important context.\n"
    "- Update existing memories when information changes rather than creating duplicates.\n"
    "- Do NOT save trivial or transient information (e.g., 'user asked about logs').\n\n"
    "GitHub Projects:\n"
    "You can manage GitHub Projects V2 in the Quote-ly organization.\n"
    "- github_list_projects: List all projects\n"
    "- github_get_project_items: View items with their Status, Priority, and other fields\n"
    "- github_create_draft_issue: Create a new draft item in a project\n"
    "- github_add_item_to_project: Add an existing issue/PR to a project\n"
    "- github_update_item_field: Change Status, Priority, or other fields on an item\n"
    "When updating fields, first use github_get_project_items to see available field values.\n\n"
    "Google Search:\n"
    "You have a google_search tool that can search the web.\n"
    "ONLY use this tool when the user explicitly asks you to search, google something, "
    "or look something up online. Examples: 'search for...', 'google...', 'look up...', "
    "'what's the latest on...'. Do NOT use it proactively or to answer questions you "
    "already know the answer to.\n\n"
    "Autonomous Agent & Background Tasks:\n"
    "You are a FULLY AUTONOMOUS agent. You can act independently without human prompting.\n"
    "Available tools: create_background_task, list_background_tasks, pause_background_task, "
    "resume_background_task, cancel_background_task.\n\n"
    "RESPONSE STYLE:\n"
    "- When confirming a task creation, be brief: just confirm with the task ID and schedule. "
    "Do NOT repeat back what the user asked for — they already know.\n"
    "- When executing a background task, just deliver the content directly. "
    "Do NOT explain that you are a background task, why you are running, or what your prompt was. "
    "Just give the user the result they asked for as if you are naturally doing it.\n"
    "- Example: If the task is to send a motivational quote, just send the quote. "
    "Do NOT say 'Here is your scheduled motivational quote as requested.'\n\n"
    "Task types:\n"
    "- 'recurring': Runs on a cron schedule. Use standard cron expressions:\n"
    "  '0 * * * *' = every hour, '*/30 * * * *' = every 30 min, "
    "  '0 9 * * 1' = Monday 9am UTC, '0 9 * * *' = daily 9am UTC.\n"
    "- 'one_shot': Runs once at a specific time. Provide an ISO 8601 datetime.\n\n"
    "CRITICAL -- Communication Intelligence:\n"
    "When creating tasks that involve sending messages or reminders:\n"
    "- Write the task prompt to FIRST CHECK if the action is still necessary.\n"
    "- Example prompt: 'Check if John has already reviewed PR #42 on "
    "Quote-ly/quotely-data-service. If not, send him a Teams message reminding him. "
    "If he already reviewed it, skip and report that no action was needed.'\n"
    "- The agent executing the task has full tool access (GitHub, CRM, memory, Teams messaging) "
    "to verify whether the action is still needed.\n"
    "- After sending a communication, consider creating a follow-up task to check for response.\n"
    "- Example follow-up: After reminding someone, create a one_shot task 4 hours later to "
    "check if they followed through. If not, send another reminder or escalate.\n\n"
    "Self-scheduling: You can create follow-up tasks during task execution. Use this to:\n"
    "- Check if someone responded to a message you sent\n"
    "- Verify that an action was completed after a reminder\n"
    "- Escalate if something hasn't been addressed after multiple attempts\n\n"
    "Convert user times to UTC using their timezone from the context brackets.\n"
    "ALWAYS pass conversation_id and user_email from the context brackets.\n\n"
    "Teams Messaging:\n"
    "You can send 1:1 Teams messages to team members using send_teams_message.\n"
    "Use lookup_team_member to check if someone is in the roster before messaging.\n"
    "Use list_team_members to see all known team members.\n"
    "Team members are automatically discovered when they message the bot or when the bot "
    "is installed in a team channel.\n\n"
    "AUTONOMY RULES:\n"
    "You are authorized to act independently on:\n"
    "- Sending Teams messages to team members\n"
    "- Checking infrastructure status (GCP, Cloud Run, logs, metrics)\n"
    "- Querying GitHub (PRs, issues, commits, workflows, projects)\n"
    "- Querying CRM data (read-only)\n"
    "- Creating and managing background tasks and schedules\n"
    "- Saving memories and context\n"
    "- Drafting reports and summaries\n"
    "- Following up on communications\n"
    "- Google searches when needed for task execution\n\n"
    "REQUIRE HUMAN APPROVAL (Devin Henderson or Cyrus) before:\n"
    "- Changing GCP settings or deploying services\n"
    "- Creating, closing, or merging GitHub PRs or deleting branches\n"
    "- Modifying CRM records (create/update/delete)\n"
    "- Publishing social media posts (use existing preview/approval flow)\n"
    "- Any action that modifies production infrastructure\n"
    "- Deleting any persistent data\n\n"
    "When in doubt about whether an action is destructive: ASK first.\n"
    "For read-only and communication actions: ACT first, report results.\n\n"
    "FedRAMP Compliance & OSCAL:\n"
    "VirtualDojo is pursuing FedRAMP Moderate authorization (ID: FR2615441197).\n"
    "FedRAMP 20x replaces document-heavy processes with automated, machine-readable evidence.\n"
    "61 Key Security Indicators (KSIs) for Moderate baseline; 70%+ must be automated.\n"
    "RFC-0024 mandates OSCAL machine-readable packages by September 2026.\n\n"
    "OSCAL-First Architecture:\n"
    "OSCAL JSON is the source of truth for all FedRAMP documentation.\n"
    "PDFs are rendered FROM OSCAL, not from markdown. Markdown files are legacy reference only.\n"
    "When updating FedRAMP content, always update the OSCAL package via oscal_update_control "
    "or oscal_generate_ssp. Never just edit the .md file.\n\n"
    "FedRAMP Infrastructure:\n"
    "- GCP project: virtualdojo-fedramp-prod (us-central1)\n"
    "- Cloud Run service: quotely (main API)\n"
    "- AlloyDB cluster: quotely-prod\n"
    "- KMS keyring: virtualdojo-keyring\n"
    "- Identity: Microsoft Entra ID (M365 GCC)\n"
    "- Evidence bucket: gs://virtualdojo-fedramp-evidence/\n"
    "- FedRAMP docs repo: Quote-ly/Fedramp\n\n"
    "Control Families You Can Assess:\n"
    "AC, AU, CM, CP, IA, RA, SC, SI, SR.\n"
    "Use fedramp_collect_evidence with the family code for detailed evidence.\n"
    "Use fedramp_evidence_summary for a quick dashboard.\n\n"
    "Evidence You CAN Collect Automatically (GCP):\n"
    "IAM policies, Cloud Run configs, log sinks, log retention, KMS keys, SCC findings,\n"
    "container vulnerabilities, Dependabot alerts, audit logs.\n\n"
    "Evidence You CANNOT Collect (requires manual/Microsoft tools):\n"
    "Entra ID MFA, Conditional Access, Intune compliance, Defender findings,\n"
    "personnel training records. When asked about these, explain they need manual collection\n"
    "and offer to create a reminder task.\n\n"
    "Remediation SLAs:\n"
    "Critical: 15 days | High: 30 days | Moderate: 90 days | Low: 180 days.\n"
    "Track these when reporting vulnerabilities. Flag overdue items.\n\n"
    "Audit Log Review Schedules:\n"
    "Daily: Admin activity, policy denied, deployments, auth failures, KMS/Secret access.\n"
    "Weekly: SCC findings, Dependabot alerts, access reviews.\n"
    "Monthly: Full evidence collection across all families, vulnerability summary, POA&M update.\n"
    "Quarterly: OSCAL package refresh, PDF rendering, Ongoing Authorization Report.\n\n"
    "OSCAL Workflow:\n"
    "Generate OSCAL -> Validate -> Review (Devin approves) -> Commit to GitHub -> Render PDF.\n"
    "Reference catalogs: NIST SP 800-53 Rev 5 (usnistgov/oscal-content), "
    "FedRAMP Moderate baseline (GSA/fedramp-automation). OSCAL version: 1.0.4.\n"
    "Use oscal_catalog_lookup to check what a specific control requires.\n\n"
    "FedRAMP Document Edit Rules:\n"
    "NEVER modify FedRAMP documents without Devin's explicit approval.\n"
    "Use fedramp_propose_edit to upload a draft to Teams for editing in Word.\n"
    "Devin edits, then tells you to commit. This is sensitive compliance documentation.\n"
    "Accuracy is paramount. Double-check control IDs, dates, and technical details.\n\n"
    "Code Review Against FedRAMP:\n"
    "Use fedramp_review_code to check source files against:\n"
    "SC-7 (CORS), SC-12 (hardcoded creds), CM-6 (error handling), SC-18 (XSS), AC-8 (login banner).\n"
    "NEVER say 'FedRAMP authorized' — say 'pursuing FedRAMP Moderate authorization'.\n\n"
    "Code Troubleshooting & Repo Access:\n"
    "You can sync and read source code from whitelisted GitHub repos locally.\n"
    "Tools: sync_repo, read_repo_file, search_repo_code, list_repo_files.\n\n"
    "TROUBLESHOOTING WORKFLOW:\n"
    "1. Determine which environment has the issue: production (main branch) or dev (development branch).\n"
    "2. Call sync_repo to ensure you have the latest code for that branch.\n"
    "3. Use search_repo_code to find relevant code (error messages, function names, config).\n"
    "4. Use read_repo_file to read the full source files you need.\n"
    "5. Cross-reference with logs (query_cloud_logs) and service status (list_cloud_run_services).\n\n"
    "Branch mapping:\n"
    "- Production issues: sync_repo(repo='Quote-ly/quotely-data-service', branch='main')\n"
    "- Development issues: sync_repo(repo='Quote-ly/quotely-data-service', branch='development')\n"
    "- Bot issues: sync_repo(repo='Quote-ly/SamurAI', branch='main')\n"
    "Always sync before reading code — the local copy may be stale.\n"
    "When troubleshooting, read the actual code, don't guess at what it does."
)


# Keywords that trigger the Pro model for complex reasoning
PRO_MODEL_KEYWORDS = [
    # OSCAL & FedRAMP document work
    "oscal", "generate ssp", "generate poam", "assessment results",
    "migrate", "update control", "link evidence", "validate package",
    "render pdf", "review code", "fedramp_review_code",
    "propose edit", "commit document", "fedramp document",
    "update ssp", "update the ssp", "control implementation",
    "catalog lookup", "look up control",
    # Code troubleshooting & analysis
    "troubleshoot", "debug", "why is", "root cause", "stack trace",
    "traceback", "exception", "bug", "broken", "not working",
    "investigate", "diagnose", "analyze code", "code review",
    "what's wrong", "error in", "fix the", "failing",
]


def _needs_pro_model(messages) -> bool:
    """Check if the conversation needs the Pro model for complex reasoning."""
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return False
    content = last_human.content.lower()
    return any(kw in content for kw in PRO_MODEL_KEYWORDS)


_GCP_KWARGS = dict(
    project=os.environ.get("GCP_PROJECT_ID"),
    location="global",
    vertexai=True,
)

# Lightweight model for instant acknowledgments — no tools, tiny prompt
_ack_llm = None


def _get_ack_llm():
    global _ack_llm
    if _ack_llm is None:
        _ack_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", **_GCP_KWARGS)
    return _ack_llm


async def _build_graph(user_id: str = "default"):
    """Build a LangGraph agent with user-specific CRM and memory tools."""
    llm_flash = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", **_GCP_KWARGS)
    llm_pro = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", **_GCP_KWARGS)

    # User-specific tools (always included in every call)
    user_specific_tools = [
        create_virtualdojo_tool(user_id),
        create_virtualdojo_list_tools(user_id),
    ] + create_memory_tools(user_id)

    # ToolNode needs ALL tools so it can execute whatever the LLM selected
    all_tools = ALL_TOOLS + user_specific_tools
    tool_node = ToolNode(all_tools, handle_tool_errors=True)

    async def call_model(state: MessagesState):
        messages = state["messages"]

        # Select model: Pro for OSCAL/FedRAMP doc/code review, Flash for everything else
        if _needs_pro_model(messages):
            llm = llm_pro
        else:
            llm = llm_flash

        # Dynamically select tools based on the user's message
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if last_human:
            selected_tools = _select_tool_groups(last_human.content) + user_specific_tools
        else:
            selected_tools = all_tools

        llm_with_tools = llm.bind_tools(selected_tools)

        # Build system prompt, injecting any relevant long-term memories
        system_content = SYSTEM_PROMPT
        if last_human:
            memory_context = await retrieve_relevant_memories(
                user_id, last_human.content
            )
            if memory_context:
                system_content += f"\n\n{memory_context}"

        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_content)] + messages

        return {"messages": [await llm_with_tools.ainvoke(messages)]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    checkpointer = await get_checkpointer()
    store = get_memory_store()
    return graph.compile(checkpointer=checkpointer, store=store)


# Cache of per-user graphs to avoid rebuilding on every message
_user_graphs: dict[str, object] = {}


async def _get_graph(user_id: str):
    """Get or create a LangGraph agent for a specific user."""
    if user_id not in _user_graphs:
        _user_graphs[user_id] = await _build_graph(user_id)
    return _user_graphs[user_id]


def reset_user_graph(user_id: str):
    """Reset a user's graph to pick up new tools (e.g. after OAuth)."""
    _user_graphs.pop(user_id, None)


async def inject_auth_message(user_id: str, conversation_id: str):
    """Inject a message into the conversation history confirming CRM auth succeeded."""
    graph = await _get_graph(user_id)
    config = {"configurable": {"thread_id": conversation_id, "user_id": user_id}}
    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="[SYSTEM: The user has successfully authenticated with VirtualDojo CRM. "
                    "The connection is now active. You can now use virtualdojo_crm and "
                    "virtualdojo_list_tools to access their CRM data. "
                    "Do NOT ask the user to connect again.]"
                )
            ]
        },
        config=config,
    )


def _extract_text(content) -> str:
    """Extract plain text from Gemini's content blocks."""
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


async def run_agent(
    user_message: str,
    conversation_id: str = "default",
    user_id: str = "default",
    user_name: str = "",
    user_timezone: str = "",
    user_email: str = "",
    status_callback=None,
) -> str:
    start = time.time()

    # Build context prefix so the LLM knows who it's talking to
    context_parts = []
    if user_name:
        context_parts.append(f"User: {user_name}")
    if user_email:
        context_parts.append(f"Email: {user_email}")
    context_parts.append(f"conversation_id: {conversation_id}")
    if user_timezone:
        from datetime import datetime
        import zoneinfo

        try:
            tz = zoneinfo.ZoneInfo(user_timezone)
            local_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M %Z")
            context_parts.append(
                f"Timezone: {user_timezone} (current time: {local_time})"
            )
        except Exception:
            context_parts.append(f"Timezone: {user_timezone}")

    message = user_message
    if context_parts:
        message = f"[{' | '.join(context_parts)}]\n{user_message}"

    # Fast acknowledgment via lightweight model (no tools, ~0.5s)
    if status_callback:
        try:
            ack_llm = _get_ack_llm()
            ack_response = await ack_llm.ainvoke([
                SystemMessage(content=(
                    "You are SamurAI, a DevOps assistant. The user just sent a message. "
                    "Write a single brief sentence acknowledging what they asked and that you're working on it. "
                    "Be natural and specific to their request. No emojis. No tool names. Examples: "
                    "'Let me check the production logs for you.' "
                    "'I\\'ll look into those GitHub issues.' "
                    "'Pulling up the service status now.'"
                )),
                HumanMessage(content=user_message),
            ])
            ack_text = _extract_text(ack_response.content).strip()
            if ack_text:
                await status_callback(ack_text)
        except Exception:
            pass  # Don't block the main agent if ack fails

    graph = await _get_graph(user_id)
    config = {"configurable": {"thread_id": conversation_id, "user_id": user_id}}

    # Friendly tool names for status updates
    _tool_labels = {
        "sync_repo": "Syncing repository",
        "read_repo_file": "Reading source code",
        "search_repo_code": "Searching codebase",
        "list_repo_files": "Browsing files",
        "query_cloud_logs": "Querying Cloud Logging",
        "list_cloud_run_services": "Checking Cloud Run services",
        "check_gcp_metrics": "Checking metrics",
        "github_list_prs": "Checking pull requests",
        "github_get_pr_details": "Reading PR details",
        "github_list_recent_commits": "Checking recent commits",
        "github_list_issues": "Checking GitHub issues",
        "github_get_issue_details": "Reading issue details",
        "github_create_issue": "Creating GitHub issue",
        "github_close_issue": "Closing GitHub issue",
        "github_list_workflow_runs": "Checking CI/CD workflows",
        "github_get_workflow_run_details": "Reading workflow details",
        "fedramp_collect_evidence": "Collecting FedRAMP evidence",
        "fedramp_daily_log_review": "Running audit log review",
        "fedramp_check_scc_findings": "Checking Security Command Center",
        "fedramp_evidence_summary": "Building compliance dashboard",
        "oscal_generate_ssp": "Generating OSCAL SSP",
        "oscal_validate_package": "Validating OSCAL package",
        "oscal_render_pdf": "Rendering PDF",
        "google_search": "Searching the web",
        "send_teams_message": "Sending Teams message",
        "create_background_task": "Creating background task",
        "manage_memory": "Saving to memory",
        "search_memory": "Searching memory",
        "fedramp_read_document": "Reading FedRAMP document",
        "fedramp_list_documents": "Browsing FedRAMP docs",
        "fedramp_search_documents": "Searching FedRAMP docs",
        "fedramp_propose_edit": "Proposing document edit",
        "fedramp_commit_document": "Committing to GitHub",
        "fedramp_review_code": "Reviewing code for compliance",
        "fedramp_check_log_retention": "Checking log retention",
        "fedramp_check_encryption": "Checking encryption",
        "fedramp_check_iam_compliance": "Checking IAM compliance",
        "fedramp_check_failed_logins": "Checking failed logins",
        "fedramp_check_dependabot_alerts": "Checking Dependabot alerts",
        "fedramp_poam_status": "Checking POA&M status",
        "oscal_generate_poam": "Generating OSCAL POA&M",
        "oscal_catalog_lookup": "Looking up NIST control",
        "oscal_update_control": "Updating control implementation",
        "oscal_migrate_from_markdown": "Migrating to OSCAL",
        "oscal_link_evidence": "Linking evidence",
        "oscal_generate_assessment_results": "Generating assessment results",
    }

    final_messages = []
    _sent_statuses: set[str] = set()  # Track sent labels to avoid duplicates

    async for event in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="updates",
    ):
        # event is a dict like {"agent": {"messages": [...]}} or {"tools": {"messages": [...]}}
        if "agent" in event:
            final_messages = event["agent"].get("messages", [])
            # Send status updates for tool calls (ack already sent above)
            if status_callback and final_messages:
                last_msg = final_messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_names = [tc.get("name", "") for tc in last_msg.tool_calls]
                    new_labels = []
                    for n in tool_names:
                        label = _tool_labels.get(n, n)
                        if label not in _sent_statuses:
                            _sent_statuses.add(label)
                            new_labels.append(label)
                    if new_labels:
                        status = "_" + ", ".join(new_labels) + "..._"
                        try:
                            await status_callback(status)
                        except Exception:
                            pass

        elif "tools" in event:
            final_messages = event["tools"].get("messages", [])

    elapsed = time.time() - start
    logger.info("[run_agent] user=%s elapsed=%.2fs", user_name or user_id, elapsed)

    if not final_messages:
        logger.error("[run_agent] empty messages in result for thread=%s", conversation_id)
        return "I wasn't able to generate a response. Please try again."

    response_text = _extract_text(final_messages[-1].content)

    # Background memory extraction — auto-saves facts from conversation
    try:
        executor = get_background_extractor()
        executor.submit(
            {"messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response_text},
            ]},
            config={"configurable": {"user_id": user_id}},
            after_seconds=1.0,
        )
    except Exception as e:
        logger.debug("Background memory extraction failed: %s", e)

    # Periodic persistence — flush memories to SQLite every call
    # (lightweight no-op if nothing changed)
    try:
        persist_memories()
    except Exception:
        pass

    return response_text
