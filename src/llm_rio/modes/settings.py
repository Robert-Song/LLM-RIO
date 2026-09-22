from pydantic import BaseModel, Field

from llm_rio.modes.cache_settings import CacheSettings as CacheSettings
from llm_rio.modes.kv_cached.settings import KVCachedSettings
from llm_rio.modes.queue.settings import QueueSettings
from llm_rio.modes.vllm_sleep.settings import SleepSettings


class ModeSettings(BaseModel):
    model_config = {"extra": "forbid"}
    queue: QueueSettings = Field(default_factory=QueueSettings)
    vllm_sleep: SleepSettings = Field(default_factory=SleepSettings)
    kv_cached: KVCachedSettings = Field(default_factory=KVCachedSettings)
