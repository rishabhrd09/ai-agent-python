import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load .env for local development (Vercel ignores this if file is missing)
load_dotenv()

# --- Configuration ---
SYSTEM_MESSAGE = (
    "You are a helpful note-taking assistant. "
    "You can read and write text files to help users manage their notes. "
    "Always use the /tmp/ directory for file operations. "
    "Be concise and helpful."
)

# --- Vercel-Safe Tools ---

@tool
def read_note(filepath: str) -> str:
    """Read the contents of a text file. Path must start with /tmp/."""
    # Enforce /tmp/ for Vercel read-only filesystem
    if not filepath.startswith('/tmp/'):
        filepath = os.path.join('/tmp', os.path.basename(filepath))

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"Contents of '{filepath}':\n{content}"
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found. (Note: On Vercel, files are ephemeral)."
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_note(filepath: str, content: str) -> str:
    """Write content to a text file. Path must start with /tmp/."""
    # Enforce /tmp/ for Vercel read-only filesystem
    if not filepath.startswith('/tmp/'):
        filepath = os.path.join('/tmp', os.path.basename(filepath))
        
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"

TOOLS = [read_note, write_note]

# --- Agent Logic ---

def get_agent():
    """Initialize agent with explicit API key from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
    # Pass api_key explicitly to ensure client finds it on Vercel
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)

async def run_agent(user_input: str) -> str:
    """Run the agent asynchronously."""
    try:
        # Initialize agent fresh on every request to pick up env vars correctly
        agent = get_agent()
        
        # Use ainvoke for async execution
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"Error running agent: {str(e)}"