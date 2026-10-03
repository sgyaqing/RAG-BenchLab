"""Built-in agent that fills a RAG system adapter config ("smart fill").

Pipeline (bounded, no framework):
  checking    judge-free: verify the fill LLM is reachable
  researching turn the platform hint into evidence: a docs URL is fetched
              directly; a platform name relies on the fill LLM's knowledge
  generating  LLM writes the adapter config (endpoint, headers, body template,
              extraction paths) as strict JSON
  probing     real-call the target system with an auto question (first item of
              the newest completed testset, else a default); if the answer path
              misses, one LLM path-fix against the raw response; if contexts
              are empty, one retry with a different question

Manual create/edit goes through run_simple_test instead: a single probe call.

The agent never writes code; when the system doesn't fit the declarative
adapter contract (async polling, SSE-only, signed auth) it fails with a clear
message in the log.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.core.host import host_gateway
from app.db.models import ModelConfig, RagSystemConfig
from app.db.session import get_session_factory
from app.services import model_connect, rag_client
from app.services.llm_http import chat_completion

logger = logging.getLogger(__name__)

_running: set[int] = set()

# Probe questions must be domain-agnostic: the probe's job is to trigger
# retrieval (any corpus, any system), not to get a good answer.
DEFAULT_PROBE_QUESTION = "这个知识库包含哪些内容？"
DEFAULT_PROBE_QUESTION_EN = "What topics does this knowledge base cover?"

# Sized for the thinking, not the answer: the config that comes back is ~100
# tokens, but a model left thinking spends 2656-5476 of its budget on that and
# then has nothing left to answer with (measured on deepseek-flash at the
# full 12000-char evidence, three runs, thinking on). 16384 is 3x the worst of
# those and well under the platforms' own caps (Ark reports 131072).
#
# Not larger, though a bigger cap generates nothing extra and so costs nothing:
# it is not free of consequences. A platform whose context window is small
# requires prompt + max_tokens to fit inside it, and one whose output cap is
# low answers a too-large value with a 400. 16384 is the compromise we chose
# knowing that; a customer below it will fail with that 400 rather than with a
# truncated answer.
CONFIG_MAX_TOKENS = 16384
MAX_DOC_CHARS = 12000
MAX_RESPONSE_CHARS = 4000

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

_CONFIG_SYSTEM_PROMPT = """You configure an HTTP adapter that lets our evaluation tool call a RAG system.

The adapter contract:
- Single synchronous POST to the endpoint URL (host is given; you supply the path)
- headers: a JSON object of extra headers; use the literal {{api_key}} where the API key goes
  (e.g. {"Authorization": "Bearer {{api_key}}"}); {} if none needed
- body_template: the request body JSON as a string; the user's question goes where the
  literal {{question}} placeholder sits
- answer_path / contexts_path: dot paths into the response JSON.
  Syntax: a.b.c walks objects, a.b[0].c indexes an array, a.b[].c maps an array to a list.
  contexts_path must yield the RETRIEVED reference chunks — the knowledge-base
  text segments the answer is grounded on (NOT the prompt, NOT the answer);
  null if the system does not return retrieved contexts.
- platform: dify | ragflow | fastgpt | custom (best guess)

Answer with ONLY a JSON object:
{"endpoint": "...", "headers": {...}, "body_template": "...",
 "answer_path": "...", "contexts_path": "... or null", "platform": "..."}

Important: if the API can stream (SSE), the body template MUST disable it —
e.g. include "stream": false, or "response_mode": "blocking" for Dify.
If the platform returns retrieved reference chunks ONLY when asked (some
platforms need a flag such as "detail": true or "citations": true in the
request body), enable that flag in body_template and provide the matching
contexts_path. Contexts matter for evaluation — leave contexts_path null
only when the system never returns retrieved chunks at all.
The field name carrying the question varies per platform (question, query,
input, ...) — follow the documentation or your knowledge of THIS platform;
do not blindly copy the example's field names.

If the system's API is asynchronous (create task then poll), streaming-only, or needs
request signing, answer {"unsupported": "reason"} instead."""

_PATH_FIX_PROMPT = """Given this real JSON response from a RAG system (possibly truncated), find:
- answer_path: dot path to the answer text (a non-empty string)
- contexts_path: dot path to the RETRIEVED context texts — the knowledge-base
  reference chunks the answer is grounded on (a LIST of text segments).
  This is NOT the answer, NOT the prompt/instructions sent to the model,
  NOT the conversation history. null if no retrieved reference chunks exist.
Path syntax: a.b.c walks objects, a.b[0].c indexes an array, a.b[].c maps an array to a list.
Answer with ONLY a JSON object: {"answer_path": "...", "contexts_path": "... or null"}"""

_BODY_FIX_PROMPT = """A call to a RAG system's chat endpoint failed with an API error.
Fix the request body template. Rules:
- It must be a valid JSON object containing the literal {{question}} placeholder
  where the user's question goes
- Respect the API's error message (missing/unknown fields, wrong types)
- If the API can stream, include "stream": false
- NEVER fabricate values you cannot know — especially IDs from the user's
  account (app/chat/knowledge-base IDs). A syntactically valid but invented ID
  is worse than none: remove such fields instead of inventing values
Answer with ONLY the corrected JSON body template as a string."""


class AssistError(Exception):
    """code: machine-readable failure category for client-side i18n guidance
    (notFound / authFailed / unreachable / noAnswer / unknown).
    urls: probe addresses that failed (shown in the failure log)."""

    def __init__(self, message: str, code: str = "unknown", urls: list[str] | None = None):
        super().__init__(message)
        self.code = code
        self.urls = urls or []


class ProbeHTTPError(Exception):
    """HTTP error from the target system, carrying the response body snippet
    (platforms often explain the cause there, e.g. 'appId is empty')."""

    def __init__(self, status: int, body: str):
        self.status = status
        super().__init__(f"HTTP {status}: {body[:300]}")


def _classify_error(e: Exception) -> str:
    """Map a probe exception to a user-actionable failure category."""
    if isinstance(e, ProbeHTTPError):
        if e.status == 404:
            return "notFound"
        if e.status in (401, 403):
            return "authFailed"
        return "httpError"
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 404:
            return "notFound"
        if status in (401, 403):
            return "authFailed"
        return "httpError"
    if isinstance(e, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return "unreachable"
    if isinstance(e, ApiPayloadError):
        # HTTP 200 with a business error ({"code": 102, "message": ...}) or a
        # body that is not JSON at all. Its message is either the platform's
        # own explanation or a description of what came back instead — both
        # worth showing.
        return "platformError"
    if isinstance(e, StreamResponseError):
        return "streamResponse"
    return "unknown"


class StreamResponseError(Exception):
    """The endpoint answered with an event stream, not a JSON document."""


class ApiPayloadError(Exception):
    """The endpoint's 200 response carried no usable payload: a business error
    ({"code": 102, "message": ...}) or a body that was not JSON at all. The
    message is either the platform's own words or a description of what came
    back instead, and both are meant to be read by the customer."""


def _update(config_id: int, **fields) -> None:
    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, config_id)
        if cfg is not None:
            for k, v in fields.items():
                setattr(cfg, k, v)
            db.commit()
    finally:
        db.close()


def _since(cfg: RagSystemConfig) -> float:
    """Seconds since the adapter row was created — the whole run, start to end.

    created_at is stored as naive UTC, which is what the rest of the codebase
    assumes it means. It never changes, so a config object read earlier in the
    run is still fine to measure from.
    """
    created = cfg.created_at
    if created is None:  # an object built in memory, never inserted
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - created).total_seconds(), 1)


def _log_missing_contexts(config_id: int) -> None:
    """Say so when the finished adapter has no contexts path.

    Not an error — plenty of systems never return the retrieved chunks, and
    `null` is the honest answer for them. But it is the one loss the customer
    can undo, by hand, right here in the adapter: without a contexts path the
    evaluation silently scores fewer metrics, and finding that out from the
    metric list afterwards is a long way from the setting that caused it.

    Reads the row rather than taking a config object: the steps before it
    (applying the generated config, validating the contexts path) write through
    their own sessions, so any object passed in from the start of the pipeline
    still says contexts_path is None — and this warning went out over a run
    whose probe had just extracted four chunks.
    """
    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, config_id)
        missing = cfg is not None and not cfg.contexts_path
    finally:
        db.close()
    if missing:
        _log(config_id, "noContexts", {})


def _log(config_id: int, key: str, params: dict | None = None) -> None:
    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, config_id)
        if cfg is not None:
            entries = json.loads(cfg.log_entries)
            entries.append(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "key": key,
                    "params": params or {},
                }
            )
            cfg.log_entries = json.dumps(entries, ensure_ascii=False)
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tools: web search, doc fetch, LLM call, probe
# ---------------------------------------------------------------------------


async def fetch_doc_text(url: str) -> str:
    """Fetch a docs page as plain text. Mintlify-hosted docs serve Markdown
    when the URL ends in .md — retried automatically when HTML yields little."""
    from bs4 import BeautifulSoup

    async def get(u: str) -> str:
        async with httpx.AsyncClient(
            timeout=20, headers=_UA, follow_redirects=True
        ) as client:
            resp = await client.get(host_gateway(u))
            resp.raise_for_status()
            return resp.text

    text = ""
    try:
        html = await get(url)
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    except Exception as e:
        logger.info("Doc fetch failed for %s (%s): %s", url, type(e).__name__, e)
        return ""
    if len(text) < 500 and not url.endswith(".md"):
        try:
            md = await get(url.rstrip("/") + ".md")
            if len(md) > len(text):
                text = md
        except Exception:
            pass
    return text[:MAX_DOC_CHARS]


async def _ask_llm(cfg: ModelConfig, system: str, user: str) -> tuple[str, dict]:
    """One chat completion. Returns (content, usage)."""
    # What the probe measured for this model when the config was saved, sent
    # verbatim — the platform's own spelling of its thinking switch is in
    # there. Calling the model as the config says is the whole point of having
    # probed it; the hardcoded "if it looks like Ark, send thinking:disabled"
    # that used to be here was the guess the probe replaced, and it could only
    # disagree with what the config actually recorded.
    #
    # Not a speed tune-up: thinking tokens come out of the same max_tokens
    # budget as the answer. deepseek-flash without this field spent thousands
    # of tokens on thinking (10-25 s), and when the budget ran out the reply
    # carried no content at all — which reached the customer as "LLM did not
    # return a JSON object: ''". With it: 89-99 tokens, 0.8 s.
    extra = json.loads(cfg.thinking_param) if cfg.thinking_param else {}
    async with httpx.AsyncClient(timeout=120) as client:
        reply = await chat_completion(
            client,
            api_format=cfg.api_format,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            messages=[{"role": "user", "content": user}],
            system=system,
            max_tokens=CONFIG_MAX_TOKENS,
            extra=extra,
            close_connection=True,
        )
        reply.raise_for_status()
        if not (reply.text or "").strip():
            # An empty reply is not something to retry. The JSON-repair round
            # downstream works by showing the model its own previous answer and
            # asking for clean JSON — with an empty answer there is nothing to
            # show, and the model, asked to correct nothing, returns nothing
            # again (measured: a second empty reply 18 s later). Raising here
            # keeps that wasted call from happening at all, and lets the log
            # name the cause instead of "did not return a JSON object: ''".
            raise AssistError("The model returned no content", code="noOutput")
        return reply.text, reply.usage


def _parse_json_object(text: str) -> dict:
    """Extract the first JSON object from LLM output (tolerates code fences)."""
    start = text.find("{")
    if start == -1:
        raise AssistError(
            f"LLM did not return a JSON object: {text[:200]!r}", code="noConfig"
        )
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise AssistError(
            f"LLM returned invalid JSON ({e}): {text[start:start + 300]!r}",
            code="noConfig",
        ) from e
    if not isinstance(obj, dict):
        raise AssistError(
            f"LLM did not return a JSON object: {text[:200]!r}", code="noConfig"
        )
    return obj


def _probe_question(lang: str = "") -> str:
    """A fixed, domain-agnostic question about the knowledge base itself, in
    the UI's language (it shows up in the configuration log).

    (Picking a question from the newest testset was fragile: no testset ->
    arbitrary default; many testsets -> the newest may be a different domain
    from the target system and fail to trigger retrieval.)"""
    return DEFAULT_PROBE_QUESTION_EN if lang == "en" else DEFAULT_PROBE_QUESTION


async def _probe(cfg: RagSystemConfig, question: str,
                 url: str | None = None) -> tuple[dict, float]:
    """Call the target system once; returns (raw JSON response, seconds)."""
    body = rag_client.render_body(cfg.body_template, question)
    headers = {"Content-Type": "application/json", "Connection": "close"}
    headers.update(rag_client.render_headers(json.loads(cfg.headers or "{}"), cfg.api_key))
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=cfg.timeout) as client:
        resp = await client.post(
            host_gateway(url or cfg.base_url), headers=headers, json=body
        )
        if resp.status_code >= 400:
            # Keep the response body: platforms often explain the cause there.
            raise ProbeHTTPError(resp.status_code, resp.text)
        if "text/event-stream" in resp.headers.get("content-type", ""):
            raise StreamResponseError(url or cfg.base_url)
        try:
            data = resp.json()
        except ValueError:
            # An HTML error page, a plain-text body, an empty one. What came
            # back instead is the only thing that says what happened.
            raise ApiPayloadError(f"response was not JSON: {resp.text[:200]}")
    # Many platforms (RAGFlow, Dify...) return error payloads with HTTP 200:
    # {"code": <non-zero-or-string>, "message": "..."}. Success payloads from
    # these APIs use code 0 or omit the field entirely.
    code = data.get("code") if isinstance(data, dict) else None
    if isinstance(code, str) or (isinstance(code, (int, float)) and code not in (0, 200)):
        raise ApiPayloadError(str(data.get("message", f"code={code}"))[:300])
    return data, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


async def run_assist(config_id: int) -> None:
    """Smart fill: research -> generate -> probe -> verify. Terminal state always."""
    if config_id in _running:
        return
    _running.add(config_id)
    try:
        await _assist_pipeline(config_id)
    except Exception as e:
        # An AssistError is already a sentence meant to be read (it reaches
        # the customer through {error} on the codes that show it); anything
        # else keeps its class name for the server log.
        detail = str(e) if isinstance(e, AssistError) else f"{type(e).__name__}: {e}"
        logger.exception("Adapter assist %s failed", config_id)
        _update(config_id, status=RagSystemConfig.STATUS_FAILED, error=detail[:1000],
                base_url="", api_key=None)  # endpoint/key are not persisted
        code = getattr(e, "code", None) or _classify_error(e)
        _log(config_id, "failed", {
            "error": detail[:500], "code": code,
            "urls": ", ".join(getattr(e, "urls", None) or []),
        })
    finally:
        _running.discard(config_id)


_DOCS_URL_PROMPT = """You know the official documentation sites of RAG platforms.
Given a platform name, answer with ONLY the URL of its official API
documentation page for the chat/completion endpoint (the page documenting how
to call the chat API). If you do not know it with confidence, answer UNKNOWN."""


async def _generate_config(
    llm_cfg: ModelConfig, host: str, hint: str, evidence: str,
    has_api_key: bool, note: str = "",
) -> dict:
    """One LLM config-generation round (with a single JSON repair retry).
    Returns the normalized generated config dict."""
    example = rag_client.PRESETS["dify"]
    user_msg = (
        f"RAG system host: {host}\nPlatform hint: {hint or '(none)'}\n"
        f"API key provided: {'yes' if has_api_key else 'no'}\n\n"
        f"Worked example (Dify): endpoint /v1/chat-messages, "
        f"headers {json.dumps(example['headers'])}, "
        f"body_template {example['body_template']!r}, "
        f"answer_path {example['answer_path']}, "
        f"contexts_path {example['contexts_path']}\n\n"
        f"Documentation (may be empty, then use your own knowledge):\n{evidence}"
    )
    if note:
        user_msg = f"{note}\n\n" + user_msg
    content, usage = await _ask_llm(llm_cfg, _CONFIG_SYSTEM_PROMPT, user_msg)
    try:
        generated = _parse_json_object(content)
    except AssistError:
        # One repair round: feed the broken output back, demand clean JSON.
        logger.info("LLM config output was not valid JSON, retrying with a "
                    "repair request. Raw output: %s", content[:500])
        repair_msg = (
            "Your previous answer was not valid JSON. Return ONLY the corrected "
            "JSON object — no code fences, no commentary.\n\n" + content[:2000]
        )
        content, usage = await _ask_llm(llm_cfg, _CONFIG_SYSTEM_PROMPT, repair_msg)
        generated = _parse_json_object(content)
    if generated.get("unsupported"):
        # The prompt offers this escape for two different situations — an API
        # that cannot be adapted (async, streaming-only, signed) and a hint
        # whose documentation carried no API spec — and the model's reason is
        # not a reliable way to tell them apart. The log line names the docs
        # case and tells the customer to check the address, which is the cause
        # far more often; the model's own words stay in the run's error field.
        raise AssistError(
            f"Unsupported system: {generated['unsupported']}", code="noDocs"
        )
    for field in ("endpoint", "body_template", "answer_path"):
        if not generated.get(field):
            # The raw output goes to the server log only. The error text itself
            # is shown to the user in the adapter log dialog, and a few hundred
            # characters of model output there is neither readable nor
            # actionable for them — but without it a failure like this one can
            # only be diagnosed by guessing.
            logger.warning(
                "LLM config output is missing %r. Raw output: %s", field, content[:500]
            )
            raise AssistError(f"LLM config missing field: {field}", code="noConfig")
    # Normalize: body_template must be a JSON *string*; models sometimes emit
    # a nested object instead.
    if isinstance(generated["body_template"], (dict, list)):
        generated["body_template"] = json.dumps(
            generated["body_template"], ensure_ascii=False)
    if isinstance(generated.get("headers"), str):
        generated["headers"] = json.loads(generated["headers"])
    return generated


def _apply_generated(config_id: int, generated: dict) -> None:
    _update(config_id, **{
        "headers": json.dumps(generated.get("headers") or {}, ensure_ascii=False),
        "body_template": str(generated["body_template"]),
        "answer_path": str(generated["answer_path"]),
        "contexts_path": generated.get("contexts_path") or None,
        "platform": str(generated.get("platform") or "custom"),
    })


async def _build_candidates(
    config_id: int, host: str, hint: str, generated: dict
) -> tuple[list[str], bool]:
    """Probe URL candidates: the user's address as-is first (when it carries a
    path), then the generated endpoint. Returns (candidates, user_had_path)."""
    from urllib.parse import urlparse

    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, config_id)
        base_url, api_key = cfg.base_url, cfg.api_key
    finally:
        db.close()

    generated_endpoint = str(generated["endpoint"])
    generated_url = (
        generated_endpoint
        if generated_endpoint.startswith("http")
        else host + "/" + generated_endpoint.lstrip("/")
    )
    candidates = []
    user_had_path = bool(urlparse(base_url).path.strip("/"))
    if user_had_path:
        candidates.append(base_url)
    # RAGFlow embeds the chat_id in the path: with a bare host + API key,
    # list the assistants and complete the URL when there is exactly one.
    if not user_had_path and "ragflow" in (
        f"{hint} {generated.get('platform', '')}".lower()
    ):
        chat_url = await _ragflow_chat_url(config_id, host, api_key)
        if chat_url and chat_url not in candidates:
            candidates.append(chat_url)
    if generated_url not in candidates:
        candidates.append(generated_url)
    return candidates, user_had_path


_CONFIG_SNAPSHOT_FIELDS = (
    "headers", "body_template", "answer_path", "contexts_path", "platform", "base_url",
)


async def _docs_rescue(
    config_id: int, llm_cfg: ModelConfig, host: str, hint: str, *,
    require_contexts: bool,
) -> bool:
    """One rescue round when internal knowledge was not enough: ask the fill
    LLM for the platform's official API-docs URL, fetch it, regenerate the
    config and re-probe. With require_contexts (the existing config already
    works but lacks contexts) the rescue is accepted only when strictly
    better — probes OK AND yields contexts; otherwise the previous config is
    restored. Never raises: failure just keeps the previous state."""
    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, config_id)
        snapshot = {f: getattr(cfg, f) for f in _CONFIG_SNAPSHOT_FIELDS}
        snapshot_key = bool(cfg.api_key)
    finally:
        db.close()

    # Ask the fill LLM for the official docs URL, with one corrective round:
    # a fetched page is accepted only if it actually looks like THIS platform's
    # docs (the LLM sometimes names the wrong site — e.g. FastGPT -> fastai).
    evidence = ""
    docs_url = ""
    feedback = ""
    tried_urls: list[str] = []
    token = hint.split()[0].lower() if hint.split() else hint.lower()
    for _ in range(2):
        prompt_user = f"Platform: {hint}" + (f"\n{feedback}" if feedback else "")
        content, _ = await _ask_llm(llm_cfg, _DOCS_URL_PROMPT, prompt_user)
        m = re.search(r"https?://[^\s)\"'>]+", content)
        if not m:
            # The model's own words are the only clue here; the other two exits
            # from this loop already name the URL they rejected.
            logger.info("Adapter %s docs rescue: fill LLM did not name a docs URL. Raw: %s",
                        config_id, content[:300])
            return False
        url = m.group(0).rstrip(".,;")
        if url in tried_urls:
            logger.info("Adapter %s docs rescue: LLM repeated %s, giving up",
                        config_id, url)
            return False
        tried_urls.append(url)
        try:
            text = await fetch_doc_text(url)
        except Exception as e:
            logger.info("Adapter %s docs rescue: fetch failed for %s: %s",
                        config_id, url, e)
            text = ""
        if text and (not token or token in text.lower()):
            evidence = text
            docs_url = url
            break
        logger.info("Adapter %s docs rescue: %s does not look like %s docs",
                    config_id, url, hint)
        feedback = (
            f"The URL {url} is not the official API documentation of {hint} "
            f"(page missing or unrelated). Give a different URL, or UNKNOWN."
        )
    if not evidence:
        return False
    logger.info("Adapter %s docs rescue: fetched %s (%d chars)",
                config_id, docs_url, len(evidence))

    note = (
        "The previous configuration probed successfully but the response "
        "contained no retrieved reference chunks. Enable the flag that makes "
        "the API return reference chunks (e.g. a detail/citations field) and "
        "provide the matching contexts_path."
        if require_contexts else
        "The previous configuration failed to probe. Fix it using the "
        "documentation below."
    )
    try:
        generated = await _generate_config(llm_cfg, host, hint, evidence,
                                           bool(snapshot_key), note=note)
    except AssistError as e:
        logger.info("Adapter %s docs rescue: regeneration failed: %s", config_id, e)
        return False
    _apply_generated(config_id, generated)
    candidates, _ = await _build_candidates(config_id, host, hint, generated)
    try:
        n_contexts = await _probe_and_fix(config_id, candidates)
    except AssistError as e:
        logger.info("Adapter %s docs rescue: probe failed: %s", config_id, e)
        if require_contexts:
            _update(config_id, **snapshot)
        return False
    if require_contexts and n_contexts == 0:
        # Not strictly better: keep the previous working config.
        _update(config_id, **snapshot)
        logger.info("Adapter %s docs rescue: still no contexts, keeping previous config",
                    config_id)
        return False
    return True


async def _assist_pipeline(config_id: int) -> None:
    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, config_id)
        if cfg is None:
            return
        llm_cfg = db.get(ModelConfig, cfg.llm_config_id) if cfg.llm_config_id else None
        hint = cfg.platform_hint or ""
        from urllib.parse import urlparse as _up

        _parsed = _up(cfg.base_url)
        host = f"{_parsed.scheme}://{_parsed.netloc}"  # scheme+host only, no path
    finally:
        db.close()
    if llm_cfg is None:
        raise AssistError("Smart-fill LLM config not found", code="llmMissing")

    # --- checking (instant, invisible): fill LLM reachable
    ok, msg, ms = await model_connect.test_connectivity(
        "llm", llm_cfg.api_format, llm_cfg.base_url, llm_cfg.api_key, llm_cfg.model
    )
    if not ok:
        raise AssistError(f"Smart-fill LLM unreachable: {msg}", code="llmUnreachable")
    _log(config_id, "connectivityOk", {"ms": ms, "llm": llm_cfg.name})
    _update(config_id, stage="researching", progress=5)

    # --- researching: hint -> evidence text
    evidence = ""
    hint_is_url = bool(re.match(r"https?://", hint))
    if hint_is_url:
        evidence = await fetch_doc_text(hint)
        _log(config_id, "docFetched", {"url": hint, "chars": len(evidence)})
        if not evidence:
            raise AssistError(
                f"Could not fetch the documentation page: {hint}", code="docFetchFailed"
            )
    # With no URL, the generation below runs on the fill LLM's own knowledge.
    # (Web search was removed: in practice it only ever surfaced marketing
    # homepages — users who have docs paste the URL directly.)
    _update(config_id, stage="generating", progress=35)

    # --- generating: LLM writes the adapter config
    generated = await _generate_config(llm_cfg, host, hint, evidence, bool(cfg.api_key))
    if not hint_is_url:
        # Logged after the generation it reports, not before it. The entry is a
        # result — "there were no docs, the model's own knowledge stood in for
        # them" — so its timestamp has to be when that result existed, the same
        # way docFetched is stamped after the fetch it reports. Written where
        # the decision was taken, it read as a promise and was dated minutes
        # before the work it described.
        _log(config_id, "internalKnowledge", {"hint": hint})
    _apply_generated(config_id, generated)
    _update(config_id, stage="probing", progress=60)

    # --- probing
    candidates, user_had_path = await _build_candidates(config_id, host, hint, generated)
    probe_error: AssistError | None = None
    n_contexts = -1
    try:
        n_contexts = await _probe_and_fix(config_id, candidates)
    except AssistError as e:
        probe_error = e

    # --- docs rescue (one round, name hints only): probe failed, or probe
    # succeeded but yielded no contexts. Ask the fill LLM for the platform's
    # official API-docs URL, fetch the real docs, regenerate and re-probe.
    # With a working config the rescue is accepted only when strictly better.
    if (probe_error is not None or n_contexts == 0) and not hint_is_url and hint:
        if await _docs_rescue(config_id, llm_cfg, host, hint,
                              require_contexts=probe_error is None):
            probe_error = None

    if probe_error is not None:
        if not user_had_path:
            # Bare host didn't work: the path may embed an app ID the agent
            # cannot know. Leave an actionable, translated guide in the log.
            _log(config_id, "guideFullUrl", {})
        raise probe_error
    _update(
        config_id,
        status=RagSystemConfig.STATUS_COMPLETED,
        progress=100,
        stage="done",
        completed_at=datetime.now(timezone.utc),
        base_url="",  # sample endpoint served its purpose; not persisted
        api_key=None,
    )
    _log_missing_contexts(config_id)
    _log(config_id, "done", {"seconds": _since(cfg)})


def _try_disable_stream(config_id: int, cfg: RagSystemConfig) -> bool:
    """The endpoint answered with SSE: inject "stream": false into the body
    template (persisted, so the user sees it). Returns True when modified."""
    try:
        body = json.loads(cfg.body_template)
    except Exception:
        return False
    if not isinstance(body, dict) or body.get("stream") is False:
        return False
    body["stream"] = False
    cfg.body_template = json.dumps(body, ensure_ascii=False)
    _update(config_id, body_template=cfg.body_template)
    return True


_CONTEXT_KEY_RE = re.compile(r"chunk|reference|retriev|citation|source|quote", re.I)
_CONTEXT_KEY_WEAK_RE = re.compile(r"context|segment|document", re.I)
_NOT_CONTEXT_RE = re.compile(r"prompt|history|question|query|instruction|message|answer", re.I)
# Value keys that are IDs/metadata, never chunk text (e.g. doc_aggs[].doc_id,
# FastGPT quoteList[].sourceId). Covers both snake_case and camelCase IDs.
_IDISH_KEY_RE = re.compile(
    r"^(id|.*_id|.*Id|ids|uuid|url|score|similarity|positions?|count|total)$", re.I)

# Values that are file paths/URLs/IDs, never chunk text (e.g. FastGPT's
# sourceId "dataset/<id>/file.md"). Paths are single tokens with separators
# or extensions; real prose (even CJK, which has no spaces) has punctuation.
_PATHISH_VALUE_RE = re.compile(r"^[\w\-./\\:~]+$")


def _contexts_look_real(values: list[str]) -> bool:
    """Reject context lists whose values mostly look like paths/URLs/IDs —
    content validation that needs no ground truth."""
    if not values:
        return False
    pathish = sum(
        1 for v in values
        if _PATHISH_VALUE_RE.match(v) and ("/" in v or "\\" in v or "." in v)
    )
    return pathish <= len(values) / 2


def _walk_context_lists(node, path: str = "") -> list[tuple[str, list[str]]]:
    """Yield (path, values) for every list-of-strings in the JSON tree."""
    found: list[tuple[str, list[str]]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            found.extend(_walk_context_lists(v, f"{path}.{k}" if path else k))
    elif isinstance(node, list) and node:
        if all(isinstance(x, str) and x.strip() for x in node):
            found.append((path + "[]", list(node)))
        elif all(isinstance(x, dict) for x in node):
            for k in node[0]:
                vals = [x.get(k) for x in node[:20]]
                if all(isinstance(v, str) and v.strip() for v in vals):
                    found.append((f"{path}[].{k}", [x[k] for x in node]))
            for x in node[:3]:
                found.extend(_walk_context_lists(x, path + "[]"))
    return found


def _find_contexts_path(data: dict) -> str | None:
    """Deterministically locate the retrieved-chunks path in a raw response:
    a list of texts under a context-ish key, never prompt/answer-ish keys,
    never id-ish value keys."""
    best: tuple[int, int, str] | None = None
    for path, values in _walk_context_lists(data):
        if _NOT_CONTEXT_RE.search(path):
            continue
        value_key = path.split("[].")[-1].split(".")[-1]
        if _IDISH_KEY_RE.match(value_key):
            continue
        if not _contexts_look_real(values):
            continue
        score = 3 if value_key.lower() in ("content", "text") else 0
        if _CONTEXT_KEY_RE.search(path):
            score += 2
        elif _CONTEXT_KEY_WEAK_RE.search(path):
            score += 1
        if score == 0:
            continue
        candidate = (score, len(values), path)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best else None


def _validate_contexts_path(config_id: int, cfg: RagSystemConfig, data: dict) -> None:
    """Keep the contexts path honest (create/smart-fill only — never on edits):
    - empty (LLM found none / user left blank): try to discover one from the
      raw response and fill it
    - set but implausible (single value, non-context key — weak LLMs pick
      data.prompt): re-locate deterministically"""
    if not cfg.contexts_path:
        discovered = _find_contexts_path(data)
        if discovered:
            cfg.contexts_path = discovered
            _update(config_id, contexts_path=discovered)
        return
    contexts = rag_client.extract_contexts(data, cfg.contexts_path)
    if (contexts and _contexts_look_real(contexts)
            and (len(contexts) > 1 or _CONTEXT_KEY_RE.search(cfg.contexts_path))):
        return
    better = _find_contexts_path(data)
    if better != cfg.contexts_path:
        cfg.contexts_path = better
        _update(config_id, contexts_path=better)


async def _probe_json(config_id: int, cfg: RagSystemConfig, question: str,
                      url: str | None = None) -> tuple[dict, float]:
    """Probe expecting JSON; on an SSE answer, add stream:false and retry once."""
    try:
        return await _probe(cfg, question, url)
    except StreamResponseError:
        if not _try_disable_stream(config_id, cfg):
            raise
        return await _probe(cfg, question, url)


async def _fix_paths(cfg: RagSystemConfig, data: dict) -> str | None:
    """One LLM path-fix round against the raw response. Returns the extracted
    answer on success, None otherwise. Persists the fixed paths."""
    db = get_session_factory()()
    try:
        llm_cfg = db.get(ModelConfig, cfg.llm_config_id)
    finally:
        db.close()
    content, usage = await _ask_llm(
        llm_cfg, _PATH_FIX_PROMPT, json.dumps(data, ensure_ascii=False)[:MAX_RESPONSE_CHARS]
    )
    fixed = _parse_json_object(content)
    if not fixed.get("answer_path"):
        # Server log only: the caller turns this into a failed adapter, and the
        # raw output is the one thing that says why.
        logger.info("Adapter %s path fix: no answer_path in the output. Raw: %s",
                    cfg.id, content[:500])
        return None
    answer = rag_client.extract_answer(data, str(fixed["answer_path"]))
    if answer is None:
        logger.info("Adapter %s path fix: %r yields no answer. Raw: %s",
                    cfg.id, fixed["answer_path"], content[:500])
        return None
    _update(cfg.id, answer_path=str(fixed["answer_path"]),
            contexts_path=fixed.get("contexts_path") or None)
    cfg.answer_path = str(fixed["answer_path"])
    cfg.contexts_path = fixed.get("contexts_path") or None
    return answer


async def _fix_body(cfg: RagSystemConfig, hint: str, error: str) -> bool:
    """One LLM body-template-fix round using the API's error message.
    Returns True when a usable template was produced and persisted."""
    db = get_session_factory()()
    try:
        llm_cfg = db.get(ModelConfig, cfg.llm_config_id)
    finally:
        db.close()
    user_msg = (
        f"Endpoint: {cfg.base_url}\nPlatform hint: {hint or '(unknown)'}\n"
        f"Current body template: {cfg.body_template}\n"
        f"API error: {error}"
    )
    content, usage = await _ask_llm(llm_cfg, _BODY_FIX_PROMPT, user_msg)
    candidate = content.strip()
    start = candidate.find("{")
    if start != -1:
        candidate = candidate[start:candidate.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            logger.info("Adapter %s body fix: output is not a JSON object. Raw: %s",
                        cfg.id, content[:500])
            return False
        template = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        logger.info("Adapter %s body fix: output is not JSON. Raw: %s",
                    cfg.id, content[:500])
        return False
    if rag_client.QUESTION_PLACEHOLDER not in template:
        logger.info("Adapter %s body fix: template has no %s placeholder. Raw: %s",
                    cfg.id, rag_client.QUESTION_PLACEHOLDER, content[:500])
        return False
    cfg.body_template = template
    _update(cfg.id, body_template=template)
    return True


async def _ragflow_chat_url(config_id: int, host: str, api_key: str | None) -> str | None:
    """RAGFlow embeds the chat_id in the endpoint path. With a bare host plus
    an API key, list the assistants and complete the URL when exactly one
    exists (several -> the user must pick; zero -> nothing to call)."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{host_gateway(host)}/api/v1/chats",
                headers={"Authorization": f"Bearer {api_key or ''}"},
            )
            resp.raise_for_status()
            data = resp.json().get("data")
    except Exception as e:
        logger.info("RAGFlow chat listing failed for %s: %s", host, e)
        return None
    # Newer versions paginate as {"chats": [...], "total": N}; older return a list.
    chats = data.get("chats", []) if isinstance(data, dict) else (data or [])
    if len(chats) == 1:
        return f"{host}/api/v1/chats/{chats[0]['id']}/completions"
    return None


async def _probe_with_body_fix(config_id: int, cfg: RagSystemConfig,
                               question: str, url: str) -> tuple[dict, float]:
    """Probe a candidate; on error payloads, one LLM body-fix round, then one
    extra retry for transient server-side errors (e.g. the target's embedding
    service restarting) before giving the candidate up. HTTP 400/422 (input
    validation) also get the body-fix round: those responses usually carry a
    machine-readable reason (e.g. FastGPT's zodError) the fix can act on."""
    try:
        return await _probe_json(config_id, cfg, question, url)
    except ProbeHTTPError as e:
        if e.status not in (400, 422):
            raise
        logger.info("Adapter %s probe miss %s: %s", config_id, url, str(e)[:200])
        if not await _fix_body(cfg, cfg.platform_hint or "", str(e)):
            raise
        return await _probe_json(config_id, cfg, question, url)
    except ApiPayloadError as e:
        logger.info("Adapter %s probe miss %s: %s", config_id, url, str(e)[:200])
        if not await _fix_body(cfg, cfg.platform_hint or "", str(e)):
            raise
        try:
            return await _probe_json(config_id, cfg, question, url)
        except ApiPayloadError as e2:
            logger.info("Adapter %s probe miss %s: %s", config_id, url, str(e2)[:200])
            await asyncio.sleep(2)
            return await _probe_json(config_id, cfg, question, url)


async def _probe_and_fix(config_id: int, candidates: list[str]) -> int:
    """Probe candidate URLs in order: the user's address as-is first (when it
    already carries a path), then the generated endpoint. A candidate wins when
    the call succeeds AND the answer can be extracted (with one path-fix round).
    Returns the number of extracted contexts (0 = none found)."""
    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, config_id)
    finally:
        db.close()
    question = _probe_question(cfg.lang)
    answer = None
    data = None
    last_error = "no candidates"
    last_code = "unknown"
    tried: list[str] = []
    for url in candidates:
        tried.append(url)
        try:
            data, elapsed = await _probe_with_body_fix(config_id, cfg, question, url)
        except Exception as e:
            # Our own exceptions already carry a sentence written for a
            # reader. Anything else keeps its class name — for an unexpected
            # failure that is the only clue there is.
            last_error = (
                str(e)
                if isinstance(e, (ProbeHTTPError, ApiPayloadError, StreamResponseError))
                else f"{type(e).__name__}: {e}"
            )
            last_code = _classify_error(e)
            logger.info("Adapter %s probe miss %s: %s", config_id, url, last_error[:200])
            data = None
            continue
        logger.info("Adapter %s probe ok %s (%d ms)", config_id, url, round(elapsed * 1000))
        answer = rag_client.extract_answer(data, cfg.answer_path)
        if answer is None:
            answer = await _fix_paths(cfg, data)
            if answer is None:
                last_error = "answer not found in the response"
                last_code = "noAnswer"
                logger.info("Adapter %s probe miss %s: %s", config_id, url, last_error)
                data = None
                continue
        _update(config_id, base_url=url)
        cfg.base_url = url
        break
    if data is None or answer is None:
        # No "Probe failed:" prefix: the log line already reads 配置失败：, and
        # the detail reaches the customer through {error} on codes that show it.
        raise AssistError(last_error, code=last_code, urls=tried)

    # No "the question didn't trigger retrieval, try another one" retry here.
    # There was one, guarded on `not contexts and cfg.contexts_path`, and it
    # could never run: _validate_contexts_path above clears the path to None
    # whenever the response carries no contexts, so the guard was false in
    # every case it was meant to catch, and the contexts it wanted to repair
    # were the empty ones it had already given up on. A response with no
    # contexts at all is what the docs rescue downstream is for.
    _validate_contexts_path(config_id, cfg, data)
    contexts = rag_client.extract_contexts(data, cfg.contexts_path)
    _update(config_id, progress=90)
    _log(config_id, "probeResult", {
        "question": question,
        "answer": (answer or "")[:100],
        "contexts": len(contexts),
    })
    return len(contexts)


async def run_simple_test(config_id: int, fill_blank_contexts: bool = True) -> None:
    """Manual create/edit: save already happened; verify with a single probe.
    fill_blank_contexts=False on edits: a deliberately cleared contexts path
    must stay cleared (auto-discovery is a create-time convenience)."""
    if config_id in _running:
        return
    _running.add(config_id)
    try:
        db = get_session_factory()()
        try:
            cfg = db.get(RagSystemConfig, config_id)
            if cfg is None:
                return
        finally:
            db.close()
        _update(config_id, status=RagSystemConfig.STATUS_CONFIGURING,
                stage="testing", progress=5, error=None)
        try:
            data, elapsed = await _probe_json(config_id, cfg, _probe_question(cfg.lang))
            answer = rag_client.extract_answer(data, cfg.answer_path)
            if answer is None:
                raise AssistError(
                    f"answer not found at path '{cfg.answer_path}' in the response",
                    code="noAnswer",
                )
            # Contexts auto-discovery/rewrite runs on CREATE only. On edits
            # the user's config is probed exactly as given — the simple test
            # never rewrites anything.
            if fill_blank_contexts:
                _validate_contexts_path(config_id, cfg, data)
            contexts = rag_client.extract_contexts(data, cfg.contexts_path)
            _update(
                config_id,
                status=RagSystemConfig.STATUS_COMPLETED,
                progress=100,
                stage="done",
                completed_at=datetime.now(timezone.utc),
                base_url="",  # connectivity test only; not persisted
                api_key=None,
            )
            _log(config_id, "probeResult", {
                "question": _probe_question(cfg.lang),
                "answer": answer[:100],
                "contexts": len(contexts),
                "ms": round(elapsed * 1000),
            })
            _log_missing_contexts(config_id)
            _log(config_id, "done", {"seconds": _since(cfg)})
        except Exception as e:
            # An AssistError is already a sentence meant to be read (it reaches
            # the customer through {error} on the codes that show it); anything
            # else keeps its class name for the server log.
            detail = str(e) if isinstance(e, AssistError) else f"{type(e).__name__}: {e}"
            _update(config_id, status=RagSystemConfig.STATUS_FAILED,
                    stage="testing", error=detail[:1000],
                    base_url="", api_key=None)
            code = getattr(e, "code", None) or _classify_error(e)
            _log(config_id, "failed", {
                "error": detail[:500], "code": code, "urls": cfg.base_url,
            })
    finally:
        _running.discard(config_id)


async def resume_interrupted() -> None:
    """Re-run adapters that were configuring when the server stopped.

    Which operation to re-run comes from the stage, not from llm_config_id: a
    smart-created adapter keeps that id for good, so dispatching on it re-ran
    the full agent for an interrupted manual EDIT and let _apply_generated
    overwrite the user's own templates — the one thing an edit must never do.
    A resumed edit also keeps fill_blank_contexts=False, since a deliberately
    cleared contexts path is part of what the user saved.
    """
    db = get_session_factory()()
    try:
        rows = db.execute(
            select(RagSystemConfig).where(
                RagSystemConfig.status == RagSystemConfig.STATUS_CONFIGURING
            )
        ).scalars().all()
        plan = [(r.id, r.stage or "", bool(r.llm_config_id)) for r in rows]
    finally:
        db.close()
    import asyncio

    for config_id, stage, has_llm in plan:
        logger.info("Resuming interrupted adapter config %s (stage=%s)", config_id, stage)
        _log(config_id, "resumed")
        if stage == "testing":
            # a manual create or edit was being verified: probe only
            asyncio.create_task(run_simple_test(config_id, fill_blank_contexts=False))
        elif stage in ("researching", "generating", "probing") or has_llm:
            asyncio.create_task(run_assist(config_id))
        else:
            asyncio.create_task(run_simple_test(config_id))
