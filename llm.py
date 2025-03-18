# llm.py
from langchain_community.chat_models import ChatOpenAI

def get_llm(api_key: str, temperature: float = 0.0):
    """Returns a configured ChatOpenAI instance."""
    return ChatOpenAI(api_key=api_key, temperature=temperature)