"""Microsoft Teams bot entrypoint — runs on Cloud Run via aiohttp."""

import os

from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import Activity

from agent import run_agent

settings = BotFrameworkAdapterSettings(
    app_id=os.environ.get("MICROSOFT_APP_ID", ""),
    app_password=os.environ.get("MICROSOFT_APP_PASSWORD", ""),
)
adapter = BotFrameworkAdapter(settings)


async def on_error(context: TurnContext, error: Exception):
    print(f"[on_turn_error] {error}", flush=True)
    await context.send_activity("Sorry, something went wrong. Please try again.")


adapter.on_turn_error = on_error


async def on_message(turn_context: TurnContext):
    user_message = turn_context.activity.text
    if not user_message:
        return

    # Send a typing indicator so Teams knows we're working
    await turn_context.send_activity(Activity(type="typing"))

    response = await run_agent(user_message)
    await turn_context.send_activity(Activity(type="message", text=response))


async def messages(req: web.Request) -> web.Response:
    if "application/json" not in req.headers.get("Content-Type", ""):
        return web.Response(status=415)

    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    await adapter.process_activity(activity, auth_header, on_message)
    return web.Response(status=200)


async def health(req: web.Request) -> web.Response:
    return web.Response(text="ok")


app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/health", health)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting bot on port {port}", flush=True)
    web.run_app(app, host="0.0.0.0", port=port)
