from contextlib import contextmanager
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

_openai_client = None

def get_openai_client():
    """Get or create the OpenAI client singleton."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        _openai_client = OpenAI(api_key=api_key) if api_key else OpenAI()
    return _openai_client

@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    from database import SessionLocal
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
