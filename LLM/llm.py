from langchain_openai import AzureChatOpenAI
from pathlib import Path
import os

try:
    # Prefer python-dotenv for consistent loading in any context
    from dotenv import load_dotenv, find_dotenv

    # Try to find .env from current cwd; if not found, look near this file
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        # Fallback: .env next to this file (works if debugger changes cwd)
        env_path = Path(__file__).resolve().parent / ".env"

    load_dotenv(dotenv_path=env_path, override=False)  # set override=True if you want .env to win
except Exception as e:
    # Optional: log but don’t crash
    print(f"Warning: could not load .env: {e}")



AZURE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_API_VERSION=os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_DEPLOYMENT=os.getenv('AZURE_OPENAI_DEPLOYMENT')


# Initialize Azure LLM WITHOUT structured output forcing
llm = AzureChatOpenAI(
    azure_endpoint= AZURE_ENDPOINT,
    openai_api_version=AZURE_API_VERSION,  # Your current version works fine
    azure_deployment=AZURE_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    openai_api_type="azure",
    temperature=0,
)