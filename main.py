from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agent import run_agent

app = FastAPI()

# Ensure you have a folder named 'templates' containing index.html
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/agent", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest):
    """
    Invoke the AI agent with a prompt.
    """
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # CRITICAL FIX: 'await' the async run_agent function
        result = await run_agent(request.prompt)
        
        return AgentResponse(response=result)
    
    except Exception as e:
        # Log error to console for Vercel logs
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error invoking agent: {str(e)}")