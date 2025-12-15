from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()  # Load environment variables from .env file


llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
response = llm.invoke("Hello, Claude! How are you today?")
print(response)