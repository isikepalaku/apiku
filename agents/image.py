from typing import Optional

from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.knowledge.agent import AgentKnowledge
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.pgvector import PgVector, SearchType

from agents.settings import agent_settings
from db.session import db_url

geo_agent_storage = PgAgentStorage(table_name="geo_agent_sessions", db_url=db_url)
geo_agent_knowledge = AgentKnowledge(
    vector_db=PgVector(table_name="geo_agent_knowledge", db_url=db_url, search_type=SearchType.hybrid)
)

def get_geo_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:

    return Agent(
        name="Geo Image Agent",
        agent_id="geo-image-agent",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        description="You are an AI agent specialized in analyzing images and providing geographical and historical context.",
        instructions=[
            "Analyze images thoroughly to identify landmarks, architectural features, and geographical locations.",
            "Provide historical context and background information about identified locations.",
            "Use web search to find recent news or updates about the locations shown in images.",
            "Cite reliable sources when providing information.",
            "If an image is not provided, inform the user that an image is required for analysis.",
        ],
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        storage=geo_agent_storage,
        read_chat_history=True,
        knowledge=geo_agent_knowledge,
        search_knowledge=True,
        monitoring=True,
        debug_mode=debug_mode,
    )
