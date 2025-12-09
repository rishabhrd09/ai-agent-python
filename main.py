from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
# Import run_agent from your agent.py file
from agent import run_agent

# --- FastAPI Setup ---
app = FastAPI()
# Ensure the templates folder is created at the root of your project
templates = Jinja2Templates(directory="templates")

# --- Models ---
class AgentRequest(BaseModel):
    """Request model for agent invocation."""
    prompt: str

class AgentResponse(BaseModel):
    """Response model for agent invocation."""
    response: str
    

# --- Routes ---

@app.get("/")
async def home(request: Request):
    """Serve the main HTML interface."""
    # This looks for index.html inside the 'templates' directory
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """
    Invoke the AI agent with a prompt.
    """
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # 🛑 FIX: Use 'await' to wait for the asynchronous run_agent function to complete 🛑
        result = await run_agent(request.prompt)
        
        return AgentResponse(response=result)
    
    except Exception as e:
        # Log the error for better debugging in Vercel logs
        print(f"Error during agent invocation: {e}") 
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")

# Note: You do not need uvicorn.run() here since Vercel handles starting the application.