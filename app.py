"""Microsoft Teams bot entrypoint — runs on Cloud Run via aiohttp."""

import asyncio
import json
import os

from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    CardFactory,
    TurnContext,
)
from botbuilder.schema import Activity

from agent import run_agent, inject_auth_message
from tools.virtualdojo_mcp import exchange_code, start_oauth_flow
from tools.social_media import _pending_cards
from tools.background_tasks import _pending_task_context
from cards.social import (
    build_social_preview_card,
    build_scheduled_posts_cards,
)
from cards.actions import (
    handle_card_action,
    handle_schedule_date_reply,
    is_awaiting_schedule_date,
    store_card_activity_id,
)

# Store conversation references for proactive messaging after OAuth
# Key: OAuth state parameter, Value: ConversationReference
_oauth_conversation_refs: dict[str, object] = {}

settings = BotFrameworkAdapterSettings(
    app_id=os.environ.get("MICROSOFT_APP_ID", ""),
    app_password=os.environ.get("MICROSOFT_APP_PASSWORD", ""),
    channel_auth_tenant=os.environ.get("MICROSOFT_APP_TENANT_ID", ""),
)
adapter = BotFrameworkAdapter(settings)


async def on_error(context: TurnContext, error: Exception):
    import traceback
    traceback.print_exc()
    print(f"[on_turn_error] {error}", flush=True)
    await context.send_activity("Sorry, something went wrong. Please try again.")


adapter.on_turn_error = on_error


async def _keep_typing(turn_context: TurnContext, stop_event: asyncio.Event):
    """Send typing indicators every 2 seconds until the stop event is set."""
    while not stop_event.is_set():
        try:
            await turn_context.send_activity(Activity(type="typing"))
        except Exception:
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


async def on_message(turn_context: TurnContext):
    # Handle Adaptive Card Action.Submit callbacks (buttons clicked)
    if turn_context.activity.value and isinstance(turn_context.activity.value, dict):
        await handle_card_action(turn_context, turn_context.activity.value)
        return

    user_message = turn_context.activity.text
    if not user_message:
        return

    conversation_id = turn_context.activity.conversation.id

    # If we're waiting for a schedule date, intercept the reply
    if is_awaiting_schedule_date(conversation_id):
        await handle_schedule_date_reply(turn_context, conversation_id, user_message)
        return

    # Start continuous typing indicator
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(turn_context, stop_typing))

    user_id = turn_context.activity.from_property.id
    user_name = turn_context.activity.from_property.name or "unknown"
    local_tz = getattr(turn_context.activity, "local_timezone", None) or ""
    # Extract user email via Teams-specific roster API
    user_email = ""
    try:
        from botbuilder.core.teams import TeamsInfo

        member = await TeamsInfo.get_member(turn_context, user_id)
        user_email = member.email or member.user_principal_name or ""
    except Exception as e:
        print(f"[on_message] TeamsInfo.get_member failed: {e}", flush=True)
    if not user_email and user_name and "@" in user_name:
        user_email = user_name
    print(f"[on_message] user={user_name} email={user_email} id={user_id}", flush=True)

    # Only allow virtualdojo.com users
    if not user_email or not user_email.lower().endswith("@virtualdojo.com"):
        stop_typing.set()
        await typing_task
        await turn_context.send_activity(
            Activity(
                type="message",
                text="Sorry, SamurAI is only available to VirtualDojo team members.",
            )
        )
        print(f"[on_message] BLOCKED unauthorized user: {user_email or user_id}", flush=True)
        return

    # Persist conversation reference for proactive messaging (background tasks)
    try:
        from task_store import get_task_store

        _store = await get_task_store()
        conv_ref = TurnContext.get_conversation_reference(turn_context.activity)
        await _store.save_conversation_ref(
            conversation_id=conversation_id,
            user_id=user_id,
            ref_json=json.dumps(conv_ref.serialize()),
        )
        # Auto-populate team roster with current user
        if user_email:
            service_url = turn_context.activity.service_url or ""
            tenant_id = ""
            if hasattr(turn_context.activity, "channel_data") and turn_context.activity.channel_data:
                tenant_id = turn_context.activity.channel_data.get("tenant", {}).get("id", "")
            await _store.save_team_member(
                email=user_email,
                teams_id=user_id,
                display_name=user_name,
                service_url=service_url,
                tenant_id=tenant_id,
            )
        # Discover other team members if in a team context
        try:
            from botbuilder.core.teams import TeamsInfo as _TeamsInfo

            members = await _TeamsInfo.get_members(turn_context)
            for m in members:
                m_email = m.email or m.user_principal_name or ""
                if m_email:
                    await _store.save_team_member(
                        email=m_email,
                        teams_id=m.id,
                        display_name=m.name or "",
                        service_url=turn_context.activity.service_url or "",
                        tenant_id=tenant_id,
                    )
        except Exception:
            pass  # Not in a team context or roster fetch failed
    except Exception as e:
        print(f"[on_message] persist conversation ref failed: {e}", flush=True)

    try:
        # Check if user is asking to connect to VirtualDojo CRM
        msg_lower = user_message.lower()
        if any(
            phrase in msg_lower
            for phrase in [
                "connect to virtualdojo",
                "connect to the virtualdojo",
                "connect virtualdojo",
                "sign in to virtualdojo",
                "sign into virtualdojo",
                "login to virtualdojo",
                "login virtualdojo",
                "connect crm",
                "login crm",
                "sign in crm",
                "connect to crm",
                "virtualdojo login",
                "virtualdojo connect",
                "virtualdojo sign in",
            ]
        ):
            login_url, oauth_state = await start_oauth_flow(user_id)
            # Save conversation reference so we can notify after OAuth completes
            conv_ref = TurnContext.get_conversation_reference(
                turn_context.activity
            )
            _oauth_conversation_refs[oauth_state] = {
                "conv_ref": conv_ref,
                "user_id": user_id,
                "conversation_id": conversation_id,
            }
            stop_typing.set()
            await typing_task
            await turn_context.send_activity(
                Activity(
                    type="message",
                    text=f"[Sign in to VirtualDojo CRM]({login_url})\n\n"
                    f"Click the link above to authenticate. Once done, "
                    f"I'll be able to access your CRM data.",
                )
            )
            return

        # Provide user context for background task tools
        _pending_task_context[conversation_id] = {
            "user_id": user_id,
            "user_name": user_name,
            "user_timezone": local_tz,
        }
        response = await run_agent(
            user_message,
            conversation_id=conversation_id,
            user_id=user_id,
            user_name=user_name,
            user_timezone=local_tz,
            user_email=user_email,
        )
        _pending_task_context.pop(conversation_id, None)
    finally:
        stop_typing.set()
        await typing_task

    # If the agent says user needs to sign in, generate a login link
    if "not signed in to VirtualDojo" in response or "sign-in link" in response:
        login_url, oauth_state = await start_oauth_flow(user_id)
        conv_ref = TurnContext.get_conversation_reference(
            turn_context.activity
        )
        _oauth_conversation_refs[oauth_state] = {
            "conv_ref": conv_ref,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        response = (
            f"You need to sign in to VirtualDojo to access CRM data.\n\n"
            f"[Sign in to VirtualDojo]({login_url})\n\n"
            f"Click the link above, then try your request again."
        )

    # Check if any tool stored card data for this conversation
    card_data = _pending_cards.pop(conversation_id, None)

    if card_data:
        card_type = card_data.get("card_type")
        await _send_card_response(
            turn_context, card_type, card_data, response, conversation_id
        )
    else:
        await turn_context.send_activity(Activity(type="message", text=response))


async def _send_card_response(
    turn_context: TurnContext,
    card_type: str,
    card_data: dict,
    text_fallback: str,
    conversation_id: str,
):
    """Send an Adaptive Card based on tool card data."""
    if card_type == "social_preview":
        card = build_social_preview_card(
            text=card_data["text"],
            platforms=card_data["platforms"],
            conversation_id=card_data["conversation_id"],
            image_url=card_data.get("image_url", ""),
            scheduled_date=card_data.get("scheduled_date", ""),
        )
        attachment = CardFactory.adaptive_card(card)
        resource = await turn_context.send_activity(
            Activity(type="message", attachments=[attachment])
        )
        # Store activity ID so we can update the card on approve/reject
        if resource and resource.id:
            store_card_activity_id(conversation_id, resource.id)

    elif card_type == "scheduled_posts":
        cards = build_scheduled_posts_cards(card_data.get("posts", []))
        if cards:
            attachments = [CardFactory.adaptive_card(c) for c in cards]
            await turn_context.send_activity(
                Activity(
                    type="message",
                    attachment_layout="carousel",
                    attachments=attachments,
                )
            )
        else:
            await turn_context.send_activity(
                Activity(type="message", text=text_fallback)
            )

    else:
        # Unknown card type — fall back to text
        await turn_context.send_activity(Activity(type="message", text=text_fallback))


async def messages(req: web.Request) -> web.Response:
    if "application/json" not in req.headers.get("Content-Type", ""):
        return web.Response(status=415)

    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    await adapter.process_activity(activity, auth_header, on_message)
    return web.Response(status=200)


async def oauth_callback(req: web.Request) -> web.Response:
    """Handle OAuth redirect from VirtualDojo after user authenticates."""
    code = req.query.get("code")
    state = req.query.get("state")
    error = req.query.get("error")

    if error:
        return web.Response(
            text=f"<html><body><h2>Authentication failed</h2><p>{error}</p>"
            f"<p>You can close this window and try again in Teams.</p></body></html>",
            content_type="text/html",
        )

    if not code or not state:
        return web.Response(
            text="<html><body><h2>Missing parameters</h2>"
            "<p>Invalid callback. Please try signing in again from Teams.</p></body></html>",
            content_type="text/html",
        )

    tokens = await exchange_code(code, state)
    if tokens:
        # Send proactive message to Teams and inject auth into conversation history
        oauth_ctx = _oauth_conversation_refs.pop(state, None)
        if oauth_ctx:
            conv_ref = oauth_ctx["conv_ref"]
            oauth_user_id = oauth_ctx["user_id"]
            oauth_conv_id = oauth_ctx["conversation_id"]

            # Inject auth confirmation into LangGraph conversation history
            try:
                await inject_auth_message(oauth_user_id, oauth_conv_id)
            except Exception as e:
                print(f"[oauth] inject_auth_message failed: {e}", flush=True)

            # Send proactive message to Teams
            try:

                async def _notify(turn_context: TurnContext):
                    await turn_context.send_activity(
                        Activity(
                            type="message",
                            text="Connected to VirtualDojo CRM! "
                            "I can now access your contacts, accounts, "
                            "opportunities, and compliance records. "
                            "What would you like to look up?",
                        )
                    )

                await adapter.continue_conversation(
                    conv_ref, _notify, settings.app_id
                )
            except Exception as e:
                print(f"[oauth] Proactive message failed: {e}", flush=True)

        return web.Response(
            text="<html><body><h2>Connected to VirtualDojo!</h2>"
            "<p>You can close this window and return to Teams. "
            "SamurAI can now access your CRM data.</p></body></html>",
            content_type="text/html",
        )
    else:
        _oauth_conversation_refs.pop(state, None)
        return web.Response(
            text="<html><body><h2>Authentication failed</h2>"
            "<p>Could not exchange the authorization code. "
            "Please try signing in again from Teams.</p></body></html>",
            content_type="text/html",
        )


async def health(req: web.Request) -> web.Response:
    return web.Response(text="ok")


app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/api/oauth/callback", oauth_callback)
app.router.add_get("/health", health)


async def on_startup(app_instance):
    from scheduler import init_scheduler

    await init_scheduler(adapter, settings.app_id)
    print("[startup] Background task scheduler started", flush=True)


async def on_cleanup(app_instance):
    from scheduler import shutdown_scheduler

    await shutdown_scheduler()
    print("[cleanup] Background task scheduler stopped", flush=True)


app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting bot on port {port}", flush=True)
    web.run_app(app, host="0.0.0.0", port=port)
