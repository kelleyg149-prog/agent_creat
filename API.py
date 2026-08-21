from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent / ".env"

load_dotenv(env_path)

api_key = os.getenv("MY_API_KEY")

print(api_key)