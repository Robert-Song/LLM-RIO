from pydantic import BaseModel, Field


class QueueSettings(BaseModel):
    model_config = {"extra": "forbid"}
    scale_window_seconds: float = Field(default=30.0, gt=0)
    minimum_marginal_efficiency: float = Field(default=0.05, ge=0, le=1)
