from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import asyncio

# Load dotenv only for local testing environment
load_dotenv()

# --- Configuration Constants ---
SYSTEM_MESSAGE = (
    "You are a helpful note-taking assistant. "
    "You can read and write text files to help users manage their notes. "
    "Always use the /tmp/ directory for file operations. "
    "Be concise and helpful."
)

# --- Tool Definitions (Vercel-Safe) ---

@tool
def read_note(filepath: str) -> str:
    """
    Read the contents of a text file from the Vercel temporary storage. 
    NOTE: The file path must start with /tmp/ or it will be corrected.
    """
    # ENFORCE VERCEL /tmp/ PATH: Corrects paths like "notes.txt" to "/tmp/notes.txt"
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
    This will overwrite the file if it exists. NOTE: The file path must start with /tmp/ or it will be corrected.
    """
    # ENFORCE VERCEL /tmp/ PATH: Corrects paths like "notes.txt" to "/tmp/notes.txt"
    if not filepath.startswith('/tmp/'):
        filepath = os.path.join('/tmp', os.path.basename(filepath))
        
    try:
        # Create the directory if it doesn't exist (useful if path includes subfolders)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{filepath}'."
    except Exception as e:
        return f"Error writing file: {str(e)}"

# --- Agent Initialization ---
TOOLS = [read_note, write_note]

def initialize_agent():
    """Initializes and returns the LangGraph agent instance, explicitly passing the API key."""
    
    # --- 🛑 THE CRITICAL FIX 🛑 ---
    # 1. Read the environment variable directly using os.environ.get()
    openai_key = os.environ.get("OPENAI_API_KEY")

    # 2. Pass the key directly to the ChatOpenAI client
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_key)
    
    # Check if the key was found; if not, raise a clear error (optional but good practice)
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable not found.")

    agent_instance = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)
    return agent_instance

async def run_agent(user_input: str) -> str:
    """Run the agent with a user query and return the response."""
    
    # Initialize agent on demand for serverless safety
    agent_instance = initialize_agent()
    
    try:
        # Use .ainvoke() for asynchronous calling
        result = await agent_instance.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1].content
    except Exception as e:
        if "API_KEY" in str(e) or "authentication" in str(e).lower():
             return "Configuration Error: The OpenAI API Key is missing or invalid. Please check Vercel Environment Variables."
        return f"Agent Runtime Error: {str(e)}"