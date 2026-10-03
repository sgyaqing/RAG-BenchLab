"""Generic HTTP adapter for calling a target RAG system.

Contract: given a question, return (answer, contexts). The adapter renders a
request-body template with the question, POSTs it, and extracts the answer and
retrieved contexts from the JSON response via dot paths.

Path syntax: segments separated by '.', e.g. "data.reference.chunks[].content".
- "a.b.c"      walk nested objects
- "a.b[0].c"   index into an array
- "a.b[].c"    map over an array, collecting field c of each element
- "a.b[]"      the array itself (each element must be a string)
Extraction never raises: missing keys / type mismatches yield None or [].

Presets are just pre-filled template values for well-known platforms.
"""

import json
import logging
import time

import httpx

from app.core.host import host_gateway

logger = logging.getLogger(__name__)

QUESTION_PLACEHOLDER = "{{question}}"

import re as _re


def normalize_url(url: str) -> str:
    """Users paste 'localhost:8080/...' without a scheme; default to http."""
    url = url.strip()
    if not _re.match(r"https?://", url, _re.IGNORECASE):
        url = "http://" + url
    return url

PRESETS: dict[str, dict] = {
    "dify": {
        "headers": {"Authorization": "Bearer {{api_key}}"},
        "body_template": json.dumps({
            "inputs": {},
            "query": QUESTION_PLACEHOLDER,
            "response_mode": "blocking",
            "user": "rag-benchlab",
        }, indent=2),
        "answer_path": "answer",
        "contexts_path": "metadata.retriever_resources[].content",
    },
    "ragflow": {
        # base_url should be the full completions URL including the chat id:
        # http://<host>/api/v1/chats/<chat_id>/completions
        "headers": {"Authorization": "Bearer {{api_key}}"},
        "body_template": json.dumps({
            "question": QUESTION_PLACEHOLDER,
            "stream": False,
        }, indent=2),
        "answer_path": "data.answer",
        "contexts_path": "data.reference.chunks[].content",
    },
    "fastgpt": {
        # OpenAI-compatible; detail=true asks for quote data (contexts shape
        # varies by version, so contexts_path is left for the user to fill).
        "headers": {"Authorization": "Bearer {{api_key}}"},
        "body_template": json.dumps({
            "model": "fastgpt",
            "messages": [{"role": "user", "content": QUESTION_PLACEHOLDER}],
            "stream": False,
            "detail": True,
        }, indent=2),
        "answer_path": "choices[0].message.content",
        "contexts_path": "",
    },
}


def render_body(template: str, question: str) -> dict:
    """Substitute {{question}} (JSON-escaped) into the body template."""
    escaped = json.dumps(question, ensure_ascii=False)[1:-1]  # strip outer quotes
    return json.loads(template.replace(QUESTION_PLACEHOLDER, escaped))


def render_headers(headers: dict, api_key: str | None) -> dict:
    """Substitute {{api_key}} in header values; drop the header if no key."""
    out = {}
    for k, v in headers.items():
        if "{{api_key}}" in v:
            if not api_key:
                continue
            out[k] = v.replace("{{api_key}}", api_key)
        else:
            out[k] = v
    return out


def extract_path(data, path: str):
    """Extract a value from nested JSON via a dot path. See module docstring."""
    if not path:
        return None
    current = [data]
    for segment in path.split("."):
        is_map = segment.endswith("[]")
        index = None
        if is_map:
            key = segment[:-2]
        elif segment.endswith("]") and "[" in segment:
            key, idx = segment[:-1].rsplit("[", 1)
            try:
                index = int(idx)
            except ValueError:
                # "chunks[abc]" is a typo in a user-supplied path, not a reason
                # to raise: this module promises extraction never raises, and a
                # ValueError from here surfaces as an unexplained per-item
                # failure rather than "that path is not valid".
                return [] if is_map else None
        else:
            key = segment
        nxt = []
        for item in current:
            if key:
                if not isinstance(item, dict) or key not in item:
                    continue
                item = item[key]
            if is_map:
                if isinstance(item, list):
                    nxt.extend(item)
            elif index is not None:
                if isinstance(item, list) and len(item) > index:
                    nxt.append(item[index])
            else:
                nxt.append(item)
        current = nxt
        if not current:
            return [] if is_map else None
    return current if len(current) != 1 else current[0]


def extract_answer(data, path: str) -> str | None:
    value = extract_path(data, path)
    return value if isinstance(value, str) and value.strip() else None


def extract_contexts(data, path: str | None) -> list[str]:
    if not path:
        return []
    value = extract_path(data, path)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    return []


async def call_rag_system(
    *,
    base_url: str,
    api_key: str | None,
    headers: dict,
    body_template: str,
    answer_path: str,
    contexts_path: str | None,
    timeout: int,
    question: str,
) -> tuple[str, list[str], float]:
    """Call the target RAG system once. Returns (answer, contexts, seconds).

    Raises on transport/HTTP errors and when the answer cannot be extracted;
    callers treat that as a per-item failure and keep going.
    """
    body = render_body(body_template, question)
    all_headers = {"Content-Type": "application/json", "Connection": "close"}
    all_headers.update(render_headers(headers, api_key))
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(host_gateway(base_url), headers=all_headers, json=body)
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.perf_counter() - started
    answer = extract_answer(data, answer_path)
    if answer is None:
        raise ValueError(f"answer not found at path '{answer_path}'")
    return answer, extract_contexts(data, contexts_path), elapsed
