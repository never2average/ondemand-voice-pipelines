from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    anthropic_api_key: str = ""
    deepgram_api_key: str = ""
    openai_api_key: str = ""
    default_asr_provider: str = "whisper"
    ier_target: float = 0.05
    max_improvement_iterations: int = 3
    demo_mode: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
