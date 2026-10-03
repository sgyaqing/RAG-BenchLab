"""Work out whether an endpoint thinks before it answers, and what stops it.

Every platform spells the switch differently, and a field it does not recognise
is ignored silently — the model keeps thinking and the only symptom is a run
several times slower than it should be. So this measures instead of guessing:
ask one tiny question, then ask it again with each candidate field, and decide
from what comes back rather than from what the request claimed.

Measured on the endpoints we have: a model that thinks spends 54 completion
tokens on a 4-character answer and returns a reasoning field; one that does not
spends 8. Two signals, because either alone has a blind spot — a platform may
report no reasoning field yet still think, and a chatty answer can look like
thinking by tokens alone.

The question is a small probability puzzle on purpose. "1+1=?" only triggers
thinking on some models, which would read as "does not think" and leave the
config believing there was nothing to turn off.
"""

import json
import logging

import httpx

from app.services.llm_http import ChatReply, chat_completion

logger = logging.getLogger(__name__)

PROBE_QUESTION = (
    "一个袋子里有 3 个红球、2 个蓝球。随机一次取出 2 个，两个都是红球的概率是多少？"
    "只用最简分数回答，不要解释。"
)

MAX_TOKENS = 64
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# Per protocol, best guess first. The first field the endpoint accepts and that
# measurably stops the thinking is the one we keep.
CANDIDATES: dict[str, list[dict]] = {
    "openai": [
        {"thinking": {"type": "disabled"}},
        {"reasoning_effort": "minimal"},
        {"enable_thinking": False},
        {"thinking_budget": 0},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ],
    "anthropic": [
        {"thinking": {"type": "disabled"}},
    ],
}

# What the UI shows: the first two read as 是, "unsupported" as 否, and None
# (no verdict — embeddings, the box was unticked, or the probe could not call)
# as a blank cell.
STATE_ALREADY_OFF = "already_off"
STATE_DISABLED = "disabled"
STATE_UNSUPPORTED = "unsupported"


async def _ask(
    client: httpx.AsyncClient,
    api_format: str,
    base_url: str,
    api_key: str | None,
    model: str,
    extra: dict,
) -> tuple[bool, bool]:
    """One probe call. Returns (answered, thinking)."""
    # No thinking setting is applied here, deliberately: this call exists to
    # measure the thinking, and turning it off first would read as "this model
    # never thinks" — the one verdict the probe must not invent.
    reply = await chat_completion(
        client,
        api_format=api_format,
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": PROBE_QUESTION}],
        max_tokens=MAX_TOKENS,
        extra=extra,
    )
    if reply.status_code >= 400:
        # The endpoint did not accept this field (or this call). Both are "no
        # verdict from this candidate" — try the next, do not abort the sweep.
        logger.info("Thinking probe: %s rejected %s (HTTP %s)",
                    model, sorted(extra) or "no extra field", reply.status_code)
        return False, False
    return True, _thinking_in(reply)


def _thinking_in(reply: ChatReply) -> bool:
    """Did that answer come out of a model that thought first?"""
    payload = reply.payload
    if reply.api_format == "anthropic":
        blocks = payload.get("content") or []
        if any(b.get("type") == "thinking" for b in blocks):
            return True
        completion = (payload.get("usage") or {}).get("output_tokens")
    else:
        message = ((payload.get("choices") or [{}])[0].get("message") or {})
        if message.get("reasoning_content"):
            return True
        completion = (payload.get("usage") or {}).get("completion_tokens")
    visible = reply.text
    if completion is None:
        return False
    # A one-line answer costs a handful of tokens; burning thirty or more on it
    # while barely answering is what thinking looks like from the outside.
    return completion >= 30 and completion >= 4 * max(1, len(visible) // 2)


async def probe(
    api_format: str, base_url: str, api_key: str | None, model: str
) -> tuple[str | None, dict | None]:
    """Returns (state, param) — param is the body fragment the runtime sends."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            answered, thinking = await _ask(client, api_format, base_url, api_key, model, {})
        except httpx.HTTPError as e:
            logger.warning("Thinking probe: baseline call failed for %s: %s", model, e)
            return None, None
        if not answered:
            return None, None
        if not thinking:
            logger.info("Thinking probe: %s does not think (state=%s)",
                        model, STATE_ALREADY_OFF)
            return STATE_ALREADY_OFF, None

        for param in CANDIDATES.get(api_format, []):
            try:
                answered, thinking = await _ask(
                    client, api_format, base_url, api_key, model, param
                )
            except httpx.HTTPError as e:
                logger.info("Thinking probe: %s errored on %s: %s", model, sorted(param), e)
                continue
            if answered and not thinking:
                logger.info("Thinking probe: %s disabled with %s", model, json.dumps(param))
                return STATE_DISABLED, param

    logger.warning("Thinking probe: %s thinks and none of %d fields stopped it",
                   model, len(CANDIDATES.get(api_format, [])))
    return STATE_UNSUPPORTED, None
