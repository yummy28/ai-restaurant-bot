from starlette.config import Config
from openai import OpenAI

config = Config(".env")

OPENAI_API_KEY = config("OPENAI_API_KEY", cast=str)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_AGENT_MODEL = "gpt-4o-mini"

NLTK_DATA_DIRECTORY = "nltk_data"
STANZA_RESOURCES_DIRECTORY = "stanza_resources"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
BOT_TOKEN = config("BOT_TOKEN", cast=str)
