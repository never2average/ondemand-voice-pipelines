from functools import lru_cache

from supabase import Client, create_client

from app.config import get_settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    return create_client(get_settings().supabase_url, get_settings().supabase_key)
