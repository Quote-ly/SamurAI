"""Microsoft Teams bot entrypoint — runs on Cloud Run via aiohttp."""

import asyncio
import os

from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import Activity

from agent import run_agent
from tools.virtualdojo_mcp import exchange_code, start_oauth_flow, is_user_authenticated

settings = BotFrameworkAdapterSettings(
    app_id=os.environ.get("MICROSOFT_APP_ID", ""),
    app_password=os.environ.get("MICROSOFT_APP_PASSWORD", ""),
    channel_auth_tenant=os.environ.get("MICROSOFT_APP_TENANT_ID", ""),
)
adapter = BotFrameworkAdapter(settings)


async def on_error(context: TurnContext, error: Exception):
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
    user_message = turn_context.activity.text
    if not user_message:
        return

    # Start continuous typing indicator
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(turn_context, stop_typing))

    user_id = turn_context.activity.from_property.id
    conversation_id = turn_context.activity.conversation.id
    user_name = turn_context.activity.from_property.name or "unknown"
    local_tz = getattr(turn_context.activity, "local_timezone", None) or ""

    try:
        # Check if user is asking to connect to VirtualDojo CRM
        msg_lower = user_message.lower()
        if any(phrase in msg_lower for phrase in ["connect to virtualdojo", "sign in to virtualdojo", "login to virtualdojo", "connect crm", "login crm"]):
            login_url = await start_oauth_flow(user_id)
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

        response = await run_agent(
            user_message,
            conversation_id=conversation_id,
            user_id=user_id,
            user_name=user_name,
            user_timezone=local_tz,
        )
    finally:
        stop_typing.set()
        await typing_task

    # If the agent says user needs to sign in, generate a login link
    if "not signed in to VirtualDojo" in response or "sign-in link" in response:
        login_url = await start_oauth_flow(user_id)
        response = (
            f"You need to sign in to VirtualDojo to access CRM data.\n\n"
            f"[Sign in to VirtualDojo]({login_url})\n\n"
            f"Click the link above, then try your request again."
        )

    await turn_context.send_activity(Activity(type="message", text=response))


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
        return web.Response(
            text="<html><body><h2>✅ Connected to VirtualDojo!</h2>"
                 "<p>You can close this window and return to Teams. "
                 "SamurAI can now access your CRM data.</p></body></html>",
            content_type="text/html",
        )
    else:
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting bot on port {port}", flush=True)
    web.run_app(app, host="0.0.0.0", port=port)
