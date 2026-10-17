import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_SERVICE_KEY")
    or os.getenv("SUPABASE_KEY", "")
)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

def check_config():
    missing = []
    if not SUPABASE_URL or "your-project-id" in SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY or "your-supabase" in SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    return missing
