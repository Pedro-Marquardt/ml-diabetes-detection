import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ConfigEnvs:
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")