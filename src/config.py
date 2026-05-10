from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required
    groq_api_key: str

    # Model config
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Paths
    catalog_path: str = "data/catalog.json"
    index_path: str = "data/catalog.index"
    meta_path: str = "data/catalog_meta.pkl"

    # Retrieval
    top_k_retrieve: int = 15   # how many candidates to pull from FAISS
    top_k_inject: int = 12     # how many to inject into the system prompt

    # Conversation limits
    max_turns: int = 8         # evaluator cap — hard limit from spec
    llm_max_tokens: int = 1500

    # Scraper
    scrape_delay_listing: float = 0.5   # seconds between listing pages
    scrape_delay_detail: float = 0.3    # seconds between detail pages

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()  # type: ignore[call-arg]
