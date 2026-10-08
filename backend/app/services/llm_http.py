"""How this app talks to a model endpoint over HTTP.

One definition per piece of knowledge: what path a Base URL needs, what headers
a protocol wants, what body a chat call carries, and how a reply is unpacked.
Every raw-HTTP caller — the connectivity test on the config page, the thinking
probe, the adapter's smart fill — goes through chat_completion, so no two of
them can disagree about any of it.

They disagreed once. The smart fill hand-built `base_url + "/messages"` while
the probe went through `_anthropic_url`, which adds the `/v1` these endpoints
expect. A call the probe had just proved reachable came back 401 from the same
key and the same endpoint, and read as a bad credential rather than a bad path.
"""

from dataclasses import dataclass, field

import httpx

from app.core.host import host_gateway


def _openai_url(base_url: str, path: str) -> str:
    return f"{host_gateway(base_url).rstrip('/')}/{path.lstrip('/')}"


def _anthropic_url(base_url: str, path: str) -> str:
    """Anthropic's own API is versioned in the path (`/v1/messages`), but the
    gateways are not consistent about it: DeepSeek's `/anthropic` and Ark's
    `/api/compatible` both leave the version off, so it is added when the Base
    URL does not already carry it."""
    base = host_gateway(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return f"{base}/{path.lstrip('/')}"


def _openai_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _anthropic_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


@dataclass
class ChatReply:
    """A decoded reply, kept uninterpreted.

    The status is carried rather than raised here because the callers want
    different things from it: the probe reads a 400 as "this endpoint rejected
    that field, try the next candidate", while the adapter reads it as a failed
    call. Turning it into an exception inside would take that choice away.
    """

    response: httpx.Response
    api_format: str
    _payload: dict | None = field(default=None, init=False, repr=False)

    @property
    def status_code(self) -> int:
        return self.response.status_code

    @property
    def payload(self) -> dict:
        if self._payload is None:
            self._payload = self.response.json()
        return self._payload

    @property
    def text(self) -> str:
        """The assistant's visible answer, whichever envelope it arrived in."""
        if self.api_format == "anthropic":
            return "".join(
                b.get("text", "")
                for b in self.payload.get("content") or []
                if b.get("type") == "text"
            )
        choices = self.payload.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""

    @property
    def usage(self) -> dict:
        return self.payload.get("usage") or {}

    def raise_for_status(self) -> "ChatReply":
        self.response.raise_for_status()
        return self


async def chat_completion(
    client: httpx.AsyncClient,
    *,
    api_format: str,
    base_url: str,
    api_key: str | None,
    model: str,
    messages: list[dict],
    max_tokens: int,
    system: str | None = None,
    extra: dict | None = None,
    timeout: httpx.Timeout | None = None,
    close_connection: bool = False,
) -> ChatReply:
    """POST one chat completion. Transport errors raise; an HTTP status does not.

    `messages` is the conversation without the system turn — pass the system
    prompt as `system`, because the two protocols carry it differently: OpenAI
    as a leading message, Anthropic as a top-level parameter. `extra` adds body
    fields (a thinking setting, a probe candidate). `timeout` overrides the
    client's timeouts for this one request, and is left out when unset:
    httpx reads a passed `None` as "no timeouts at all", not as "the client's
    timeouts" — the request-level argument replaces the client's wholesale.
    """
    if api_format == "anthropic":
        url = _anthropic_url(base_url, "messages")
        headers = _anthropic_headers(api_key)
        payload: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            payload["system"] = system
    else:
        url = _openai_url(base_url, "chat/completions")
        headers = _openai_headers(api_key)
        payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            payload["messages"] = [{"role": "system", "content": system}, *messages]
    if extra:
        payload.update(extra)
    if close_connection:
        headers["Connection"] = "close"
    if timeout is not None:
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)
    else:
        response = await client.post(url, headers=headers, json=payload)
    return ChatReply(response=response, api_format=api_format)
