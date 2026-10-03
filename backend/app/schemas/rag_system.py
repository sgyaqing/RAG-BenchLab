from datetime import datetime

from pydantic import BaseModel, Field


class RagSystemCreateIn(BaseModel):
    """mode=smart: agent fills the config; mode=manual: fields given directly."""

    mode: str = Field(pattern="^(smart|manual)$")
    name: str = Field(min_length=1, max_length=128)
    base_url: str = Field(min_length=1, max_length=512)
    api_key: str | None = None
    # UI locale at creation time; the probe question follows it
    lang: str | None = Field(default=None, pattern="^(zh|en)$")
    # smart
    llm_config_id: int | None = None
    platform_hint: str | None = None
    # manual
    headers: str = "{}"
    body_template: str | None = None
    answer_path: str | None = None
    contexts_path: str | None = None


class RagSystemUpdateIn(BaseModel):
    """Edit dialog: manual fields only; saving re-runs the simple test."""

    name: str = Field(min_length=1, max_length=128)
    base_url: str = Field(min_length=1, max_length=512)
    api_key: str | None = None
    lang: str | None = Field(default=None, pattern="^(zh|en)$")
    headers: str = "{}"
    body_template: str = Field(min_length=1)
    answer_path: str = Field(min_length=1, max_length=256)
    contexts_path: str | None = Field(default=None, max_length=256)


class RagSystemOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    platform: str
    base_url: str
    api_key: str | None
    headers: str
    body_template: str
    answer_path: str
    contexts_path: str | None
    llm_config_id: int | None
    llm_name: str | None
    platform_hint: str | None
    status: str
    progress: float
    stage: str
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class RagSystemPage(BaseModel):
    total: int
    items: list[RagSystemOut]


class RagSystemLogOut(BaseModel):
    entries: list[dict]
    error: str | None
