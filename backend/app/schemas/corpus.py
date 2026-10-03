import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Chinese/letters/digits/space/dash/underscore only; safe as a directory name.
NAME_PATTERN = re.compile(r"^[\w\- 一-鿿]+$", re.UNICODE)


class CorpusCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v or not NAME_PATTERN.match(v):
            raise ValueError(
                "Name may only contain letters, digits, Chinese characters, spaces, '-' and '_'"
            )
        return v


class CorpusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    status: str
    total_files: int
    processed_files: int
    success_files: int
    has_kg: bool = False  # whether data/corpus/<name>/kg.json exists
    convert_seconds: float | None
    created_at: datetime
    completed_at: datetime | None


class CorpusPage(BaseModel):
    total: int
    items: list[CorpusOut]


class NameCheckOut(BaseModel):
    available: bool


class CorpusLogOut(BaseModel):
    entries: list[dict] = []  # timestamped events, as the other three logs carry
    total_files: int
    success_files: int
    failed_files: list[str]
    convert_seconds: float | None
    completed_at: datetime | None
