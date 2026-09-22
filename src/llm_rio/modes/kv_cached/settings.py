from pydantic import Field

from llm_rio.modes.cache_settings import CacheSettings


class KVCachedSettings(CacheSettings):
    max_workers_per_gpu: int = Field(default=2, ge=1)
