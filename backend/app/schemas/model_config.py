from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelType = Literal["llm", "embedding"]
ApiFormat = Literal["openai", "anthropic"]


class ModelConfigIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    type: ModelType
    api_format: ApiFormat
    base_url: str = Field(min_length=1, max_length=512)
    api_key: str | None = Field(default=None, max_length=512)
    model: str = Field(min_length=1, max_length=256)
    # The dialog's "add the setting that turns thinking off" box, sent as-is.
    # Defaults to off here so that creating a config programmatically never
    # spends probe calls; the dialog sends its own box, which starts ticked.
    auto_disable_thinking: bool = False

    @model_validator(mode="after")
    def check_combination(self) -> "ModelConfigIn":
        if self.type == "embedding" and self.api_format == "anthropic":
            raise ValueError("Anthropic does not provide embedding models")
        return self


class ModelConfigOut(ModelConfigIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    # What the probe concluded, so the dialog can report it right after a save
    # and the list can show 是 / 否: "already_off" and "disabled" both mean
    # thinking is off, "unsupported" means it could not be turned off, and None
    # means no verdict (an embedding model, the box was unticked, or the probe
    # could not reach the endpoint).
    thinking_state: str | None = None


class ModelConfigPage(BaseModel):
    total: int
    items: list[ModelConfigOut]


class ConnectivityTestIn(BaseModel):
    type: ModelType
    api_format: ApiFormat
    base_url: str = Field(min_length=1)
    api_key: str | None = None
    model: str = Field(min_length=1)


class ConnectivityTestOut(BaseModel):
    success: bool
    message: str
    duration_ms: int | None = None


class ListModelsIn(BaseModel):
    api_format: ApiFormat
    base_url: str = Field(min_length=1)
    api_key: str | None = None


class ListModelsOut(BaseModel):
    models: list[str]
