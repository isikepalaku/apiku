from enum import Enum
from typing import Optional

from agno.agent import Agent
from utils.log import logger

from agents.tipidter_agui import get_tipidter_agui_agent


class AGUIAgentType(str, Enum):
    """Available AG-UI agent types"""
    tipidter = "tipidter"
    # Future agents can be added here
    # kuhap = "kuhap"
    # kuhp = "kuhp"
    # narkotika = "narkotika"


def get_agui_agent(
    agent_id: AGUIAgentType,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    """
    Get AG-UI agent instance by agent ID.
    
    Args:
        agent_id: The ID of the agent to retrieve
        user_id: Optional user identifier
        session_id: Optional session identifier
        debug_mode: Enable debug mode
        
    Returns:
        Agent: The requested agent instance
        
    Raises:
        ValueError: If agent_id is not supported
    """
    logger.info(f"Creating AG-UI agent: {agent_id}")
    
    if agent_id == AGUIAgentType.tipidter:
        return get_tipidter_agui_agent(
            user_id=user_id,
            session_id=session_id,
            debug_mode=debug_mode,
        )
    # Future agents can be added here
    # elif agent_id == AGUIAgentType.kuhap:
    #     return get_kuhap_agui_agent(user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    # elif agent_id == AGUIAgentType.kuhp:
    #     return get_kuhp_agui_agent(user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    # elif agent_id == AGUIAgentType.narkotika:
    #     return get_narkotika_agui_agent(user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    
    else:
        raise ValueError(f"Unsupported AG-UI agent: {agent_id}")


def get_available_agui_agents() -> list[str]:
    """
    Get list of all available AG-UI agent IDs.
    
    Returns:
        List[str]: List of available agent identifiers
    """
    return [agent.value for agent in AGUIAgentType]


def validate_agui_agent_id(agent_id: str) -> bool:
    """
    Validate if the provided agent ID is supported.
    
    Args:
        agent_id: The agent ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        AGUIAgentType(agent_id)
        return True
    except ValueError:
        return False 