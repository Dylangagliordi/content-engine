import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5433"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "qualume")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "qualume")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
