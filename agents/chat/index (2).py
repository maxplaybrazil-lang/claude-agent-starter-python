"""
Claude Agent SDK chat handler — EdgeOne Makers agent-python format.

Route: POST /chat
Response: SSE stream (text/event-stream)

SSE event protocol:
  event: text_delta  data: {"delta": "..."}
  event: tool_called data: {"tool": "ToolName"}
  event: image       data: {"imageId": "...", "base64": "...", "mimeType": "...", "size": ...}
  event: ping        data: {"ts": 1710000000000}
  event: error       data: {"message": "..."}
  event: done        data: {"stopped": false}

Session persistence:
  Uses ctx.store to save user/assistant messages for /history recovery.

Tools:
  EdgeOne platform sandbox tools (commands/files/code_interpreter/browser)
  bridged via Claude SDK's MCP Server mechanism.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncGenerator
from uuid import UUID

from dotenv import load_dotenv

load_dotenv()

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        create_sdk_mcp_server,
        query,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

from .._model import collect_gateway_env, resolve_model_name
from .._logger import create_logger
from ._stream import (
    StreamState,
    iter_query_messages,
    sanitize_assistant_text,
    sdk_message_to_sse,
    sse_event,
)


logger = create_logger("chat")
HEARTBEAT_INTERVAL_S = 5
MCP_SERVER_NAME = "edgeone"

SYSTEM_PROMPT = (
  'You are an EdgeOne Makers Claude Agent SDK (Python) starter example: an out-of-the-box Agent template that helps developers quickly run through and validate platform capabilities.\n' +
  'When introducing yourself, clearly say that you are a demo Agent built with Claude Agent SDK (Python) on EdgeOne Makers, designed to showcase tool calling, streaming responses, and session memory for developers.\n' +
  'You can use the EdgeOne platform tools listed below, plus project skills exposed by the Claude Agent SDK.\n\n' +
  'Available tools:\n' +
  '- commands: execute safe shell commands in the sandbox (e.g. date, ls, uname).\n' +
  '- files: read, write, list, makeDir, exists, and remove files inside the sandbox.\n' +
  '  Parameters: op is required; path is required for most ops; content is required for write.\n' +
  '- code_interpreter: run code in an isolated interpreter. This is a REAL Python execution environment, '
  'not a simulation — it can install and use real libraries (Pillow/PIL for images, fpdf2 or reportlab for '
  'PDFs, matplotlib for charts, etc.) and produce real binary output.\n' +
  '  Parameters: language (for example "python") and code.\n' +
  '- browser: fetch pages or interact with web pages by screenshot, click, type, or evaluate.\n' +
  '  Parameters: op is required; use url for fetch; use selector, text, or script when needed.\n\n' +
  'Available project skills:\n' +
  '- sandbox-algorithms: use this when the user asks to compute or verify deterministic algorithmic results such as Fibonacci sequences, factorials, primes, sorting, combinations, or explicitly asks for sandbox-algorithms.\n\n' +
  'Filesystem boundary:\n' +
  '- Use Claude Code Read only for project skill resources under .claude/skills, such as SKILL.md references or scripts needed by a loaded skill.\n' +
  '- Use the EdgeOne files tool for user workspace files, temporary files, generated artifacts, and all non-skill file operations.\n\n' +
  'Tool-use rules:\n' +
  '1. Use a tool only when it is necessary to answer the user concretely.\n' +
  '2. Call tools one at a time and wait for each result before deciding the next step.\n' +
  '3. Never invent, simulate, or paraphrase tool results. If a tool result is unavailable, say so.\n' +
  '4. If a tool call fails, do not repeat it blindly and do not switch to unrelated operations.\n' +
  '   Briefly explain the failure, adjust the parameters only if the fix is clear, otherwise ask the user for guidance.\n' +
  '5. Do not perform destructive file or shell operations unless the user explicitly asks for them.\n' +
  '6. If a tool returns an image or screenshot, do not include base64 strings, data:image URLs, or Markdown image links in your text. Briefly say the image is shown in the chat.\n' +
  '7. If the task can be answered without tools or skills, answer directly and keep the response concise.\n' +
  'When the user explicitly names a project skill, load that skill before doing the task.\n\n' +

  'IMPORTANT — you are NOT a text-only model in this environment:\n' +
  'You have a real code_interpreter tool that executes actual Python with real libraries. Because of this, '
  'you must NEVER tell the user that you "cannot generate images" or "cannot generate PDFs" or that you are '
  '"just a text-based AI" — that is false here and is considered a failure to use an available tool. If the '
  'user asks for an image, a PDF, or another generated file, you must attempt it with code_interpreter before '
  'concluding it is impossible. Only report failure if you actually tried the tool call and it errored.\n' +
  'Decide before you speak: never open your reply with a doubt or disclaimer about whether you can do '
  'something ("as a text-based AI...", "I\'m not sure I can...", "I don\'t have the ability to..."). If the '
  'request needs a tool, call the tool first and let the result determine what you say — don\'t narrate '
  'uncertainty and then contradict yourself by succeeding right after. Say what you did, not what you doubted.\n\n' +

  'Showing apps, sites, images, and files to the user (the chat UI renders these automatically):\n\n' +

  'WEBPAGES / APPS / UI MOCKUPS:\n' +
  '- Output ONE complete, self-contained HTML document (inline <style> and <script>, no external files) '
  'inside a single fenced code block tagged ```html ... ```. Do not also paste the code as plain prose — '
  'the fenced block IS the deliverable; just add a short sentence before/after describing it.\n' +
  '- Design quality is mandatory, not optional. Never ship browser-default styling (default serif headings, '
  'default button/input chrome, pure black text on white with no layout). Every page must include:\n' +
  '  • A real visual identity: pick ONE cohesive color palette (2-3 colors + neutrals) and stick to it — no '
  'default blue links, no unstyled <button>/<input>.\n' +
  '  • Deliberate typography: a clear font stack (system-ui or a Google Font), a real type scale (distinct '
  'sizes/weights for headings vs body), generous line-height.\n' +
  '  • Real layout: use flexbox/grid with consistent spacing (an 8px-based scale), not elements stacked with '
  'default margins. Content should have max-width and padding, not touch the viewport edges.\n' +
  '  • Polish details: rounded corners, subtle shadows/borders, hover states on interactive elements, and '
  'enough whitespace to not feel cramped.\n' +
  '  • Responsive: it must not break on a narrow (~380px) mobile viewport.\n' +
  '  • A CSS reset at the top (`*{box-sizing:border-box;margin:0;padding:0}` at minimum).\n' +
  '- Match the aesthetic to the request\'s context (e.g. a kids app ≠ a B2B dashboard ≠ a landing page) instead '
  'of reusing one generic look every time.\n\n' +

  'IMAGES (photos, illustrations, generated graphics):\n' +
  '- To create an image, use code_interpreter to actually generate real image bytes (e.g. with Pillow: draw '
  'shapes/text, or render a matplotlib figure, or any other real generation method available to you — do not '
  'just describe an image in words unless the user explicitly asks for a text description).\n' +
  '- Read the resulting file as bytes, base64-encode them, and output ONLY that base64 string (no explanation, '
  'no line breaks inside it) inside a fenced block tagged ```png (or ```jpg if that is the format you produced). '
  'Nothing else should be inside that code block — the platform decodes it into a real displayed image with a '
  'download button.\n' +
  '- If a tool already returns an image directly (e.g. a browser screenshot), do not print its raw URL or '
  'base64/data-URI as plain text — the platform already extracts and displays it. Just reference it briefly '
  'in words ("the image is shown above").\n' +
  '- If you are linking to an already-hosted external image instead of generating one, wrap the link in '
  'Markdown image syntax `![description](url)` rather than a bare URL — the platform turns that into a proper '
  'image card with a download button.\n\n' +

  'DOWNLOADABLE FILES:\n' +
  '- Plain-text formats (.md, .csv, .json, .txt, .svg, .xml, .yaml): output the content inside a fenced code '
  'block tagged with that language (```csv, ```json, ```md, etc). It becomes a download card automatically.\n' +
  '- PDF: use code_interpreter to actually generate a real PDF (e.g. with the `fpdf2` or `reportlab` Python '
  'library — install it first if it is not available), read the resulting file\'s bytes, base64-encode them, '
  'and output ONLY the base64 string (no explanation, no line breaks inside it) inside a fenced block tagged '
  '```pdf. Nothing else should be in that code block. The platform decodes it into a real downloadable PDF card.\n' +
  '- Audio/video/office files you generate as real binaries (mp3, mp4, docx, xlsx, pptx, zip): same pattern — '
  'base64-encode the real bytes and output them alone inside a fenced block tagged with that extension '
  '(```mp4, ```docx, etc).\n' +
  '- VIDEO specifically: use code_interpreter to actually render a real .mp4 — for example, build frames with '
  'Pillow/matplotlib and encode them with `imageio` (imageio.mimwrite/get_writer with an ffmpeg-backed writer) '
  'or with `moviepy`. Install the library first if it is missing (e.g. `pip install imageio[ffmpeg]` or '
  '`pip install moviepy`). If installation or encoding genuinely fails (for example no ffmpeg binary available '
  'in the sandbox), say plainly that video generation is not supported in this environment right now — do not '
  'claim success and do not paste fake base64. If it does succeed, read the resulting file\'s bytes, '
  'base64-encode them, and output ONLY that base64 string inside a fenced block tagged ```mp4, nothing else.\n' +
  '- Never fabricate a PDF/image/audio/video by writing placeholder or made-up text inside a ```pdf/```png/```mp4 '
  'block — only put real base64 bytes there, produced by an actual code_interpreter call. If the tool call '
  'genuinely fails after a real attempt, say so instead of faking the output.\n\n' +

  'Only produce one app/file per request unless the user explicitly asks for more than one.'
)


def _normalize_uuid(value: str) -> str | None:
    """Return canonical UUID string, or None if value is not a valid UUID."""
    try:
        return str(UUID(value))
    except (TypeError, ValueError):
        return None


async def resolve_claude_session_binding(
    session_store: Any,
    conversation_id: str,
) -> tuple[str | None, str | None]:
    """
    Bind Claude SDK session to frontend conversation_id.

    First request for a conversation uses session_id=<conversation_id> to create
    a deterministic SDK session. Later requests use resume=<conversation_id>
    when that transcript already exists in session_store.
    """
    session_id = _normalize_uuid(conversation_id)
    if not session_id:
        logger.log(f"[session] skip SDK session binding: invalid conversation_id={conversation_id!r}")
        return None, None

    try:
        from claude_agent_sdk._internal.sessions import project_key_for_directory

        # project_key is load-bearing: EdgeOne ClaudeSessionStore.load() uses it
        # as a namespace prefix on blob keys. Drop it and load() returns None.
        project_key = project_key_for_directory(os.getcwd())
        entries = await session_store.load({"project_key": project_key, "session_id": session_id})
        if entries:
            logger.log(f"[session] resume Claude SDK session_id={session_id}, entries={len(entries)}")
            return None, session_id
        logger.log(f"[session] create Claude SDK session_id={session_id}")
    except Exception as e:
        logger.error(f"[session] failed to inspect session_store for resume: {e}")

    return session_id, None


def build_agent_options(
    session_store=None,
    mcp_server=None,
    mcp_server_name: str = MCP_SERVER_NAME,
    allowed_tools: list[str] | None = None,
    session_id: str | None = None,
    resume: str | None = None,
) -> "ClaudeAgentOptions":
    """Build Claude Agent SDK options. EdgeOne tools come from MCP."""
    cwd = os.getcwd()
    skill_read_allow_rules = [
        "Read(.claude/skills/**)",
        f"Read({cwd}/.claude/skills/**)",
    ]
    # Merge incoming MCP tool names with the built-in Read scoping rules.
    # The Python SDK's `settings` field only accepts a JSON-file path
    # (str | None), unlike the TS SDK which also accepts an inline Settings
    # dict. Trying to pass a dict raises CLIConnectionError("Failed to start
    # Claude Code: expected str, bytes or os.PathLike object, not dict") at
    # subprocess launch. So we route the same `permissions.allow` intent
    # through `allowed_tools` instead — the CLI treats both as auto-allow
    # rules with identical syntax.
    merged_allowed_tools = list(
        dict.fromkeys((allowed_tools or []) + skill_read_allow_rules)
    )
    opts = ClaudeAgentOptions(
        model=resolve_model_name(),
        system_prompt=SYSTEM_PROMPT,
        cwd=cwd,
        # Keep Claude Code's built-in tools narrowly scoped: Skill loads
        # project skills, and Read may only access .claude/skills resources.
        # EdgeOne sandbox tools are exposed separately through MCP below.
        tools=["Skill", "Read"],
        allowed_tools=merged_allowed_tools,
        setting_sources=["project"],
        skills="all",
        permission_mode="dontAsk",
        max_turns=5,
        env=collect_gateway_env(),
        include_partial_messages=True,
        max_buffer_size=20 * 1024 * 1024,  # 20MB — enough for browser screenshots
        session_id=session_id,
        resume=resume,
    )
    if session_store is not None:
        opts.session_store = session_store
    if mcp_server is not None:
        opts.mcp_servers = {mcp_server_name: mcp_server}
    return opts


def build_prompt_with_history(user_message: str, history: list[dict[str, str]]) -> str:
    """
    Prepend recent conversation turns as plain-text context ahead of the
    user's current message. Used only as a fallback when the SDK's own
    session resume isn't available, so the model doesn't lose context and
    reintroduce itself mid-conversation.
    """
    if not history:
        return user_message
    lines = [
        "(Internal context — recent turns of this ongoing conversation, for your "
        "reference only. Do not repeat or re-summarize this block; do not "
        "reintroduce yourself. Just answer the current message below, keeping "
        "continuity with this context.)"
    ]
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append("\n(End of context. Current message:)")
    lines.append(user_message)
    return "\n".join(lines)


async def handler(ctx: Any) -> AsyncGenerator[str, None]:
    """EdgeOne Makers entry point (async generator streaming)."""
    cid = ctx.conversation_id or ""
    logger.log(f"[chat] entered with cid={cid!r}")

    body = ctx.request.body
    user_message: str = body.get("message", "") if isinstance(body, dict) else ""
    if not user_message.strip():
        yield sse_event("error", {"message": "'message' is required"})
        yield sse_event("done", {"stopped": False})
        return

    # Optional client-side history fallback (see build_prompt_with_history below):
    # the Aura frontend sends recent turns here so the model still has context
    # even if the SDK's own session_store fails to resume server-side.
    raw_history = body.get("history") if isinstance(body, dict) else None
    client_history: list[dict[str, str]] = (
        [h for h in raw_history if isinstance(h, dict) and h.get("role") and h.get("content")]
        if isinstance(raw_history, list) else []
    )

    # Extract frontend-generated message IDs for history alignment
    user_msg_id: str = body.get("userMsgId", "") if isinstance(body, dict) else ""
    bot_msg_id: str = body.get("botMsgId", "") if isinstance(body, dict) else ""

    # Extract user ID for store scoping
    raw_user_id = body.get("userId") or body.get("user_id") or "" if isinstance(body, dict) else ""
    user_id = str(raw_user_id).strip() or None

    if not _SDK_AVAILABLE:
        yield sse_event("error", {"message": "claude_agent_sdk is not installed"})
        yield sse_event("done", {"stopped": False})
        return

    cancel_signal = ctx.request.signal
    store_adapter = ctx.store

    # Get Claude session store for transcript persistence (matches TS reference).
    # This gives the SDK multi-turn context, preventing chaotic/repeated tool calls.
    try:
        raw_session_store = store_adapter.claude_session_store()
        logger.log(f"[session_store] enabled, type={type(raw_session_store).__name__}, value={raw_session_store is not None}")
    except Exception as e:
        raw_session_store = None
        logger.error(f"[session_store] failed to get claude_session_store: {e}")
    session_store = raw_session_store

    # Save user message (with frontend-generated ID if available)
    if cid:
        # === DEBUG: dump all store messages for this conversation ===
        try:
            all_msgs = await store_adapter.get_messages(conversation_id=cid, limit=100, order="asc")
            logger.log(f"[debug_store] conversation={cid}, total_messages={len(all_msgs)}")
            for m in all_msgs:
                role = getattr(m, "role", "?")
                msg_id = getattr(m, "message_id", "?")
                content = getattr(m, "content", "")
                preview = str(content)[:200] if content else ""
                created_at = getattr(m, "created_at", 0)
                logger.log(f"[debug_store]   [{role}] id={msg_id} ts={created_at} content={preview}")
        except Exception as e:
            logger.error(f"[debug_store] failed to dump: {e}")
        # === END DEBUG ===

        try:
            # append_message accepts only: conversation_id, role, content, metadata, user_id.
            # message_id is not supported (the SDK auto-generates one).
            await store_adapter.append_message(
                conversation_id=cid,
                role="user",
                content=user_message,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"[store] failed to save user message: {e}")

    # Build EdgeOne platform tools → Claude Agent SDK MCP server
    raw_tools = ctx.tools
    if not hasattr(raw_tools, "to_claude_mcp_server"):
        yield sse_event("error", {"message": "context.tools.to_claude_mcp_server is unavailable."})
        yield sse_event("done", {"stopped": False})
        return

    edgeone_mcp = raw_tools.to_claude_mcp_server(MCP_SERVER_NAME, {"always_load": True})
    logger.log("[tool_debug][mcp_server]", {
        "name": getattr(edgeone_mcp, "name", None),
        "allowed_tools": getattr(edgeone_mcp, "allowed_tools", None),
        "tools": [
            {
                "name": getattr(tool, "name", None) if not isinstance(tool, dict) else tool.get("name"),
                "description": getattr(tool, "description", None) if not isinstance(tool, dict) else tool.get("description"),
                "input_schema": getattr(tool, "input_schema", None) if not isinstance(tool, dict) else tool.get("input_schema"),
            }
            for tool in (getattr(edgeone_mcp, "tools", None) or [])
        ],
    })
    mcp_server = create_sdk_mcp_server(
        name=edgeone_mcp.name,
        tools=edgeone_mcp.tools,
    )

    sdk_session_id, sdk_resume = await resolve_claude_session_binding(session_store, cid)
    options = build_agent_options(
        session_store=session_store,
        mcp_server=mcp_server,
        mcp_server_name=edgeone_mcp.name,
        allowed_tools=edgeone_mcp.allowed_tools,
        session_id=sdk_session_id,
        resume=sdk_resume,
    )

    stopped = False
    stream_state = StreamState(bot_msg_id=bot_msg_id)

    # If the SDK isn't actually resuming an existing session (sdk_resume is
    # None), fall back to prepending the client-sent history as plain-text
    # context, so a silently-broken session_store doesn't make the model
    # forget the conversation and reintroduce itself every turn.
    effective_prompt = (
        user_message if sdk_resume is not None
        else build_prompt_with_history(user_message, client_history)
    )

    try:
        response_iter = query(prompt=effective_prompt, options=options).__aiter__()
        async for item_type, msg in iter_query_messages(response_iter, cancel_signal, HEARTBEAT_INTERVAL_S):
            if item_type == "cancelled":
                logger.log(f"[cancel] cancel_signal observed, stopping stream cid={cid!r}")
                stopped = True
                break
            if item_type == "finished":
                break
            if item_type == "ping":
                yield sse_event("ping", {"ts": int(time.time() * 1000)})
                continue

            events, should_stop = sdk_message_to_sse(msg, stream_state, logger)
            for event in events:
                yield event
            if should_stop:
                break

    except Exception as e:  # noqa: BLE001
        logger.error(f"[error] {e}")
        yield sse_event("error", {
            "message": str(e),
            "errorType": type(e).__name__,
            "detail": repr(e),
        })

    # Save assistant response (with frontend-generated ID if available)
    # Save even if text is empty but images were sent (use placeholder)
    assistant_content = sanitize_assistant_text(stream_state.full_assistant_text).strip()
    if not assistant_content and stream_state.has_images:
        assistant_content = "[image]"

    if store_adapter and cid and assistant_content:
        try:
            # append_message accepts only: conversation_id, role, content, metadata, user_id.
            await store_adapter.append_message(
                conversation_id=cid,
                role="assistant",
                content=assistant_content,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"[store] failed to save assistant response: {e}")

    yield sse_event("done", {"stopped": stopped})
