from typing import List, Optional
from logging import getLogger
from agno.app.agui.app import AGUIApp
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from agents.agui_operator import AGUIAgentType, get_agui_agent, get_available_agui_agents
logger = getLogger(__name__)

######################################################
## Router for the AG-UI Interface
######################################################

agui_router = APIRouter(prefix="/agui", tags=["AG-UI"])

# Dictionary to store AG-UI app instances
_agui_apps: dict[AGUIAgentType, AGUIApp] = {}


def create_agui_app(agent_id: AGUIAgentType) -> AGUIApp:
    """
    Create AG-UI app instance following the documentation structure.
    
    Args:
        agent_id: The ID of the agent to create AG-UI app for
        
    Returns:
        AGUIApp: The AG-UI app instance
    """
    # Setup your Agno Agent, can be any Agno Agent
    agent = get_agui_agent(agent_id)
    
    # Setup the AG-UI app sesuai dokumentasi
    agui_app = AGUIApp(
        agent=agent,
        name=f"{agent.name} AG-UI",
        app_id=f"{agent_id.value}_agui",
    )
    
    logger.info(f"Created AG-UI app for agent: {agent_id}")
    return agui_app


def get_agui_app(agent_id: AGUIAgentType) -> AGUIApp:
    """
    Get or create AG-UI app instance for the specified agent.
    
    Args:
        agent_id: The ID of the agent to get AG-UI app for
        
    Returns:
        AGUIApp: The AG-UI app instance
    """
    if agent_id not in _agui_apps:
        _agui_apps[agent_id] = create_agui_app(agent_id)
    
    return _agui_apps[agent_id]


class ChatRequest(BaseModel):
    """Request model for chat with AG-UI agent"""
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Halo, saya butuh bantuan tentang kasus pembalakan liar di hutan lindung",
                "user_id": "user123",
                "session_id": "session456",
                "stream": False
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat with AG-UI agent"""
    response: str
    status: str
    agent_id: str
    session_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Berdasarkan pencarian knowledge base, kasus pembalakan liar di hutan lindung...",
                "status": "success",
                "agent_id": "tipidter-agui",
                "session_id": "session456"
            }
        }


class AgentInfoResponse(BaseModel):
    """Response model for agent information"""
    agent_id: str
    name: str
    description: str
    status: str
    app_id: str
    endpoints: dict


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    agent_id: str
    message: str


@agui_router.get("", response_model=List[str])
async def list_agui_agents():
    """
    Returns a list of all available AG-UI agent IDs.

    Returns:
        List[str]: List of AG-UI agent identifiers
    """
    return get_available_agui_agents()


@agui_router.get("/{agent_id}/info", response_model=AgentInfoResponse)
async def get_agent_info(agent_id: AGUIAgentType):
    """
    Get information about a specific AG-UI agent.
    
    Args:
        agent_id: The ID of the agent to get info for
        
    Returns:
        AgentInfoResponse: Agent information
    """
    try:
        agui_app = get_agui_app(agent_id)
        agent = agui_app.agent
        
        return AgentInfoResponse(
            agent_id=agent.agent_id,
            name=agent.name,
            description=agent.description or "No description available",
            status="active",
            app_id=agui_app.app_id,
            endpoints={
                "agui_protocol": f"/api/v1/agui/{agent_id.value}/agui",
                "chat": f"/api/v1/agui/{agent_id.value}/chat",
                "info": f"/api/v1/agui/{agent_id.value}/info",
                "health": f"/api/v1/agui/{agent_id.value}/health"
            }
        )
    except Exception as e:
        logger.error(f"Error getting agent info for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {str(e)}"
        )


@agui_router.get("/{agent_id}/health", response_model=HealthResponse)
async def health_check(agent_id: AGUIAgentType):
    """
    Health check for a specific AG-UI agent.
    
    Args:
        agent_id: The ID of the agent to check health for
        
    Returns:
        HealthResponse: Health status
    """
    try:
        agui_app = get_agui_app(agent_id)
        return HealthResponse(
            status="healthy",
            agent_id=agui_app.agent.agent_id,
            message=f"AG-UI agent {agent_id} is running properly"
        )
    except Exception as e:
        logger.error(f"Health check failed for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent health check failed: {str(e)}"
        )


# Mount AG-UI apps for each agent
def setup_agui_routes():
    """
    Setup AG-UI routes for all available agents.
    This function mounts the AG-UI FastAPI apps as sub-applications.
    """
    for agent_type in AGUIAgentType:
        try:
            # Create AG-UI app instance
            agui_app = get_agui_app(agent_type)
            
            # Get the FastAPI app from AG-UI - this contains the /v1/agui endpoint
            agui_fastapi_app = agui_app.get_app()
            
            # Mount the AG-UI app routes under the agent-specific path
            # This exposes the AG-UI protocol endpoint at /{agent_id}/v1/agui
            agui_router.mount(
                f"/{agent_type.value}",
                agui_fastapi_app,
                name=f"agui_{agent_type.value}"
            )
            
            logger.info(f"Mounted AG-UI app for agent: {agent_type.value} at /{agent_type.value}/v1/agui")
            
        except Exception as e:
            logger.error(f"Failed to setup AG-UI routes for {agent_type.value}: {str(e)}")


# Alternative direct chat endpoint (for testing or non-AG-UI clients)
@agui_router.post("/{agent_id}/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_agui_agent(agent_id: AGUIAgentType, request: ChatRequest):
    """
    Direct chat endpoint for AG-UI agents (alternative to AG-UI protocol).
    This can be used for testing or clients that don't implement AG-UI protocol.
    
    **Example Request:**
    ```json
    {
        "message": "Halo, saya butuh bantuan tentang kasus pembalakan liar di hutan lindung",
        "user_id": "user123",
        "session_id": "session456",
        "stream": false
    }
    ```
    
    Args:
        agent_id: The ID of the agent to interact with
        request: Chat request containing message and optional parameters
        
    Returns:
        ChatResponse: Agent response with status and metadata
    """
    logger.debug(f"Direct chat request for {agent_id}: {request.dict()}")
    
    try:
        # Validate message
        if not request.message or request.message.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        # Get the agent instance with session info
        agent = get_agui_agent(
            agent_id=agent_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        
        # Run the agent
        response = await agent.arun(request.message, stream=request.stream)
        
        return ChatResponse(
            response=response.content,
            status="success",
            agent_id=agent.agent_id,
            session_id=request.session_id,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error in direct chat for {agent_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        ) 