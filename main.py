from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool  # Importing the search tool defined in tools.py

load_dotenv()  # Load environment variables from .env file

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    resources: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that wil help generate a search paper.
            Answer the user query and use neccessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder","{chat_history}"),
        ("human","{query}"),
        ("placeholder","{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools = tools
)

agents_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
query = input("What can i help you search today? ")
raw_response = agents_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Failed to parse response:", e)
    print("Raw response:", raw_response)
