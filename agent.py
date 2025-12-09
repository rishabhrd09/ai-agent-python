from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

# 1. Load dotenv only to ensure API key is available LOCALLY.
# Vercel relies on the environment variable set in the dashboard.
load_dotenv()

# --- Configuration Constants ---
SYSTEM_MESSAGE = (
    "You are a helpful note-taking assistant. "
    "You can read and write text files to help users manage their notes. "
    "Always use the /tmp/ directory for file operations. "
    "Be concise and helpful."
)
TOOLS = [] # Tools list is defined below

# --- Centralized Agent Initialization ---
# We will initialize the agent only once per request inside the function
# to ensure it uses the correct runtime environment variables.

def initialize_agent():
    """Initializes and returns the LangGraph agent."""
    # Instantiating the LLM here ensures it gets the OPENAI_API_KEY
    # from the environment (Vercel Dashboard) at runtime.
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent_instance = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)
    return agent_instance

# --- Tool Definitions (Vercel Safe) ---

@tool
def read_note(filepath: str) -> str:
    """
    Read the contents of a text file from the Vercel temporary storage. 
    NOTE: The file path must start with /tmp/.
    """
    # 2. Enforce /tmp/ path for Vercel safety
    if not filepath.startswith('/tmp/'):
        filepath = os.path.join('/tmp', os.path.basename(filepath))

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"Contents of '{filepath}':\n{content}"
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found. Did you write it first?"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_note(filepath: str, content: str) -> str:
    """
    Write content to a text file in the Vercel temporary storage. 
    This will overwrite the file if it exists. NOTE: The file path must start with /tmp/.
    """
    # 2. Enforce /tmp/ path for Vercel safety
    if not filepath.startswith('/tmp/'):
        filepath = os.path.join('/tmp', os.path.basename(filepath))
        
    try:
        # Create the directory if it doesn't exist (e.g., /tmp/notes)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"

# 3. Update the TOOLS list
TOOLS.extend([read_note, write_note])


# --- Main Runner Function (Async for FastAPI) ---

async def run_agent(user_input: str) -> str:
    """Run the agent with a user query and return the response."""
    
    # 4. Agent Initialization moved inside the function
    # The agent is created on demand for each invocation.
    try:
        agent_instance = initialize_agent()

        result = await agent_instance.ainvoke( # Using ainvoke for async compatibility with FastAPI
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1].content
    except Exception as e:
        # If the failure is due to missing API key, the error will show here.
        if "API_KEY" in str(e) or "authentication" in str(e).lower():
             return "Configuration Error: The OpenAI API Key is missing or invalid. Please check Vercel Environment Variables."
        return f"Agent Error: {str(e)}"