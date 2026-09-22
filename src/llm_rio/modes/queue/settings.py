from pydantic import BaseModel


class QueueSettings(BaseModel):
    model_config = {"extra": "forbid"}
