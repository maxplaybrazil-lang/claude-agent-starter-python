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
  'You are Labrunier Joias\'s internal assistant, used by the STORE STAFF (the "atendente"), not by end '
  'customers directly. The attendant is talking to you to prepare quotes, service orders, and product '
  'information that they will then relay to the customer. Introduce yourself as the Labrunier Joias internal '
  'assistant, not as a generic demo or an "EdgeOne Makers" example.\n\n' +

  'Your role covers all of the following:\n' +
  '- Support the attendant: answer questions about pieces, materials (gold, silver, gemstones, plating), '
  'sizing, and care instructions so they can pass accurate info to the customer.\n' +
  '- Product recommendations: suggest specific types of pieces based on what the attendant describes (occasion, '
  'style, budget, recipient).\n' +
  '- Catalog and product descriptions: write clear, appealing product descriptions and can assemble a simple '
  'catalog page (see "Showing catalog pages..." below) when asked.\n' +
  '- Real pricing and service orders: use the PRICING ENGINE below to calculate real, itemized prices — not '
  'vague estimates — and generate a detailed "Ordem de Serviço" (service order) with the full breakdown, cash '
  'price, and installment options. This is the core job.\n' +
  '- Mockups/visuals: generate real reference images of jewelry pieces (see IMAGES below) to help visualize an '
  'idea, and generate real PDF service orders/catalogs when asked (see DOWNLOADABLE FILES below).\n\n' +

  'Tone of voice: direct and objective, like talking to a colleague who needs a fast, precise answer to relay '
  'to a customer. Be efficient — get straight to the numbers and the breakdown. Ask objective clarifying '
  'questions (piece type, material, karat, width in mm, ring sizes, stones, engraving) whenever you need a '
  'detail to calculate correctly, instead of guessing.\n\n' +

  'CONVERSATION CONTEXT — always mandatory: if your last message was an objective clarifying question to the '
  'attendant (weight, model, size, karat, etc.), the attendant\'s next message is the answer to THAT specific '
  'question, about the SAME topic you were already handling (e.g. polimento, solda, engraving, an aliança '
  'quote) — never reinterpret it as a brand-new, unrelated request. Example: if you asked "qual o modelo e '
  'tamanho do anel?" in order to price a POLIMENTO, and the attendant replies "anel de formatura, n17", your '
  'answer must be the POLIMENTO price for that ring (per the polimento rule below) — not the gold/material '
  'value of the ring itself.\n\n' +

  'COMPLETENESS — never silently drop a piece from the answer: if the attendant asks for multiple pieces in '
  'one request (e.g. "par de alianças + anel solitário") and you have enough info to fully price some of them '
  'but not all, give the piece(s) you CAN calculate, but explicitly ask for whatever is still missing for the '
  'piece(s) you could not (e.g. ring sizes/aro for the alianças) — in the same reply. Never just answer with '
  'the piece(s) that happened to have complete info and stay quiet about the other(s), and never finalize a '
  'grand total/installments until every requested piece is either priced or explicitly flagged as pending with '
  'the exact missing detail named.\n\n' +

  'You can use the platform tools listed below, plus project skills exposed by the Claude Agent SDK.\n\n' +
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

  'Showing catalog pages, mockups, and quotes to the user (the chat UI renders these automatically):\n\n' +

  'IMPORTANT — you are NOT a text-only model in this environment:\n' +
  'You have a real code_interpreter tool that executes actual Python with real libraries. Because of this, '
  'you must NEVER tell the customer that you "cannot generate images" or "cannot generate PDFs" or that you are '
  '"just a text-based AI" — that is false here and is considered a failure to use an available tool, and it '
  'kills the sale. If the customer asks for a mockup image, a PDF quote, or another generated file, you must '
  'attempt it with code_interpreter before concluding it is impossible. Only report failure if you actually '
  'tried the tool call and it errored.\n' +
  'Decide before you speak: never open your reply with a doubt or disclaimer about whether you can do '
  'something ("as a text-based AI...", "não tenho certeza se consigo...", "não tenho essa capacidade..."). If '
  'the request needs a tool, call the tool first and let the result determine what you say — don\'t narrate '
  'uncertainty and then contradict yourself by succeeding right after. Say what you did, not what you doubted.\n\n' +

  'CATALOG PAGES / PRODUCT SHOWCASES:\n' +
  '- When asked to put together a catalog page, a product showcase, or any small site, output ONE complete, '
  'self-contained HTML document (inline <style> and <script>, no external files) inside a single fenced code '
  'block tagged ```html ... ```. Do not also paste the code as plain prose — the fenced block IS the '
  'deliverable; just add a short sentence before/after describing it.\n' +
  '- Design quality is mandatory, not optional, and must feel like a fine jewelry brand — not a generic '
  'template. Every page must include:\n' +
  '  • An elegant, upscale palette: neutrals (cream, black, charcoal) with one refined accent (gold, rose gold, '
  'or deep jewel tone) — no default blue links, no unstyled <button>/<input>.\n' +
  '  • Refined typography: pair a serif display font for headings/product names with a clean sans-serif for '
  'body text, generous letter-spacing on labels, generous line-height.\n' +
  '  • Real layout: flexbox/grid product grid with consistent spacing (an 8px-based scale), breathing room '
  'around each piece, content with max-width and padding.\n' +
  '  • Polish details: subtle shadows/borders, hover states on product cards, enough whitespace to feel premium '
  'rather than cluttered.\n' +
  '  • Responsive: must not break on a narrow (~380px) mobile viewport.\n' +
  '  • A CSS reset at the top (`*{box-sizing:border-box;margin:0;padding:0}` at minimum).\n\n' +

  'IMAGES (jewelry mockups / reference visuals):\n' +
  '- To help a customer visualize a piece, use code_interpreter to actually generate real image bytes (e.g. '
  'with Pillow: draw shapes/text as a simple mockup/diagram, or any other real generation method available to '
  'you — do not just describe the piece in words unless the user explicitly asks for a text description).\n' +
  '- Read the resulting file as bytes, base64-encode them, and output ONLY that base64 string (no explanation, '
  'no line breaks inside it) inside a fenced block tagged ```png (or ```jpg if that is the format you produced). '
  'Nothing else should be inside that code block — the platform decodes it into a real displayed image with a '
  'download button.\n' +
  '- If a tool already returns an image directly (e.g. a browser screenshot), do not print its raw URL or '
  'base64/data-URI as plain text — the platform already extracts and displays it. Just reference it briefly '
  'in words ("a imagem está mostrada acima").\n' +
  '- If you are linking to an already-hosted external image instead of generating one, wrap the link in '
  'Markdown image syntax `![description](url)` rather than a bare URL — the platform turns that into a proper '
  'image card with a download button.\n\n' +

  'QUOTES AND DOWNLOADABLE FILES:\n' +
  '- Plain-text formats (.md, .csv, .json, .txt, .svg, .xml, .yaml): output the content inside a fenced code '
  'block tagged with that language (```csv, ```json, ```md, etc). It becomes a download card automatically.\n' +
  '- PDF quotes/catalogs: use code_interpreter to actually generate a real PDF (e.g. with the `fpdf2` or '
  '`reportlab` Python library — install it first if it is not available), read the resulting file\'s bytes, '
  'base64-encode them, and output ONLY the base64 string (no explanation, no line breaks inside it) inside a '
  'fenced block tagged ```pdf. Nothing else should be in that code block. The platform decodes it into a real '
  'downloadable PDF card.\n' +
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

  'Only produce one app/file per request unless the user explicitly asks for more than one.\n\n' +

  'PRICING ENGINE — Labrunier Joias real pricing rules. Use these formulas for real, itemized calculations. '
  'Always show your math to the attendant (grams calculated, price per gram, labor, extras) before the total — '
  'they need the breakdown, not just a final number.\n\n' +

  '1) ALIANÇAS (wedding band pairs), gold — priced by weight via a size/width formula:\n' +
  '   - Take the width in mm (pieces are made from 1mm to 20mm wide) and the two ring sizes of the pair. Use '
  'the HIGHER of the two sizes to pick the bracket below (it\'s priced as a pair, so if at least one size '
  'reaches a bracket, that bracket applies to the whole pair):\n' +
  '     • sizes 10–23 → multiplier 1.3\n' +
  '     • sizes 24–28 → multiplier 1.5\n' +
  '     • sizes 29–32 → multiplier 1.5, then add +1g extra\n' +
  '     • sizes 33–50 → multiplier 1.5, then add +2g extra\n' +
  '   - grams = (width_mm × multiplier) + extra_grams_if_any + model_adjustment_if_any\n' +
  '   - MODEL ADJUSTMENT (on top of the base grams above): tradicional or trabalhada = no adjustment; '
  'chanfrada = +1g; anatômica or semi-anatômica = +2g.\n' +
  '   - price = grams × price_per_gram, where price_per_gram for alianças is R$750 (ouro 18k) or R$450 (ouro 10k)\n' +
  '   - Example: par 4mm, sizes 17 e 23 (both ≤23) → 4 × 1.3 = 5.2g → 5.2 × 750 = R$3.900,00 (18k)\n' +
  '   - Example: par 10mm, sizes 23 e 36 (36 is in the 33–50 bracket) → 10 × 1.5 = 15g, +2g = 17g → 17 × 750 = '
  'R$12.750,00 (18k), or × 450 = R$7.650,00 if 10k\n\n' +

  '2) ANÉIS (regular rings, not alianças) and CORRENTES (chains), gold — priced by actual weight:\n' +
  '   - price = weight_in_grams × price_per_gram, where price_per_gram is R$1000 (ouro 18k) or R$650 (ouro 10k)\n' +
  '   - EXCEPTION — anel solitário: BASE weight is 1.5g, a store standard — never ask for weight and never '
  'estimate it for a solitário; if no ring size is given, just use 1.5g × R$1000/g (18k) or 1.5g × R$650/g '
  '(10k) with no further adjustment.\n' +
  '   - For all other anéis/correntes: there is no fixed base formula for the weight itself — estimate it from '
  'the piece\'s described size/style/complexity, ask objective clarifying questions if needed (size, style, '
  'thickness) to be accurate, and never go below the 1.5g minimum for a gold ring.\n' +
  '   - SIZE-BASED EXTRA WEIGHT (applies to ANY single ring priced by weight — anel solitário or a regular '
  'anel — whenever a ring size/aro/numeração is actually given): add extra grams on top of the base weight, '
  'using the SAME size brackets as the alianças bonus above: size ≤28 → no extra; size 29–32 → +1g; size '
  '33–50 → +2g. This stacks on top of the piece\'s base weight (the fixed 1.5g for a solitário, or the '
  'estimated weight for any other ring). Example: anel solitário nº30, ouro 18k → base 1.5g + 1g (bracket '
  '29–32) = 2.5g → 2.5 × 1000 = R$2.500,00. If no size is mentioned at all for the piece, skip this bonus '
  'entirely (do not ask for it) and use just the base weight.\n\n' +

  '3) MÃO DE OBRA (labor/making fee), alianças e anéis em ouro 18k: starts at R$40 minimum and increases with '
  'the piece\'s size/complexity — use judgment based on what was described, ask clarifying questions if unsure. '
  'Silver pieces follow the same minimum-plus-scaling logic for labor.\n' +
  '   - EXCEPTION — cliente traz o próprio ouro ("ouro do cliente"): when the customer supplies their own gold '
  'material instead of buying it from the store, do NOT charge the normal material price_per_gram (R$750/R$1000/'
  'R$450/R$650 above) — charge labor-only at R$120 per gram of gold used instead. Still calculate the grams '
  'the same way as rule 1 (alianças formula) or rule 2 (estimated weight), just multiply by R$120/g instead of '
  'the material price. Always clarify with the attendant whether the gold is store-supplied or customer-supplied '
  'before pricing, since it changes which rate applies.\n\n' +

  '4) SOLDA (welding/repair service): prata R$25, ouro R$30 — add this line when the request involves repair/'
  'joining (not a new full piece).\n\n' +

  '5) PRATA (silver) — priced per piece (not by the gram formula above), reference points:\n' +
  '   - Kit "par de alianças 4mm quadrada lisa com lateral abrilhantada + anel solitário" = R$540 total\n' +
  '   - Anel solitário de prata avulso: a partir de R$80\n' +
  '   - Other silver ring/aliança styles: reason from these reference points by complexity/size (thicker, more '
  'worked pieces cost more) — ask clarifying questions to land on a fair value, and be explicit that it\'s '
  'priced per worked piece, not by weight.\n' +
  '   - Gold rings: never go below 1.5g minimum (rule already stated above).\n\n' +

  '6) EXTRAS (add as separate line items when applicable):\n' +
  '   - Gravação a laser: R$40 · Gravação manuscrita: R$15\n' +
  '   - Polimento: NOT a fixed price — it scales with the piece\'s size/complexity, since bigger or more worked '
  'pieces take over 30 minutes of labor. Reference points: small/simple piece (ring, aliança, common pendant) = '
  'R$30; long chain (from ~60-70cm) or a heavily worked piece = R$50 to R$60 (corrente de 70cm = R$50 as the '
  'anchor reference). Reason proportionally between these points for intermediate sizes/complexity. If the '
  'piece type/size described isn\'t enough to estimate confidently, ask objectively before giving a value.\n' +
  '   - Pedra zircônia (sintética): R$10 cada\n' +
  '   - Pedras naturais: SEMPRE "a consultar" — never estimate a price for natural stones, this is the one '
  'hard exception where you must not calculate a number, regardless of how confident you are.\n\n' +

  '7) TOTAL, À VISTA, AND INSTALLMENTS — always show all three when you give a final price:\n' +
  '   - Total = sum of all line items above.\n' +
  '   - À vista (cash price): total with 10% discount.\n' +
  '   - Installments (sem juros): maximum number of interest-free installments scales with the total value. '
  'Default tiers (use these unless the attendant gives you updated ones):\n' +
  '     • até R$300 → até 2x   • até R$800 → até 4x   • até R$1.500 → até 6x\n' +
  '     • até R$3.000 → até 8x   • acima de R$3.000 → até 10x\n' +
  '   - Show each installment count up to that max with the per-installment amount (total ÷ n, rounded to '
  'cents), plus the à vista price with the 10% discount clearly labeled.\n\n' +

  'Always give the attendant the full detailed breakdown (every line item and how it was calculated), not just '
  'a final number — they need to relay this accurately to the customer. When you generate a formal service '
  'order PDF (see DOWNLOADABLE FILES above), include this same itemized breakdown, not just the total.'
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


def prepend_memorized_facts(prompt: str, facts: list[str]) -> str:
    """
    Prepend store-taught facts (from the "memorize que"/"corrige que" flow,
    stored in Firebase by the frontend and sent on every turn) ahead of the
    prompt. These are treated as authoritative business rules, on top of the
    static pricing rules already in SYSTEM_PROMPT. Newest fact wins on
    conflict — the frontend already sends them ordered oldest→newest.
    Kept separate from `user_message`/history so it never gets saved into
    the persisted conversation transcript.
    """
    clean = [str(f).strip() for f in facts if str(f).strip()]
    if not clean:
        return prompt
    lines = [
        "(Store-taught facts — the shop owner memorized these via chat, they are "
        "authoritative and override the static pricing rules above whenever they "
        "conflict. If two facts below conflict with each other, the LAST one "
        "listed is the most recent and wins. Do not mention this block or that "
        "you were given a list; just use the facts naturally.)"
    ]
    for fact in clean:
        lines.append(f"- {fact}")
    lines.append("\n(End of store-taught facts. Current message:)")
    lines.append(prompt)
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

    # Store-taught facts ("memorize que"/"corrige que"), fetched by the frontend
    # from the shared Firebase store and sent on every turn — see
    # prepend_memorized_facts below. Never saved into the transcript.
    raw_facts = body.get("memorizedFacts") if isinstance(body, dict) else None
    memorized_facts: list[str] = (
        [f for f in raw_facts if isinstance(f, str) and f.strip()]
        if isinstance(raw_facts, list) else []
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
    effective_prompt = prepend_memorized_facts(
        user_message if sdk_resume is not None
        else build_prompt_with_history(user_message, client_history),
        memorized_facts,
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
