from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def read_note(filepath: str) -> str:
    """Read the contents of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"Contents of '{filepath}':\n{content}"
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_note(filepath: str, content: str) -> str:
    """Write content to a text file. This will overwrite the file if it exists."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"
    
    
TOOLS = [read_note, write_note]

SYSTEM_MESSAGE = (
    "You are a helpful note-taking assistant. "
    "You can read and write text files to help users manage their notes. "
    "Be concise and helpful."
)

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)


def run_agent(user_input: str) -> str:
    """Run the agent with a user query and return the response."""
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"
