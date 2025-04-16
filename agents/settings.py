from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """Agent settings that can be set using environment variables.

    Reference: https://pydantic-docs.helpmanual.io/usage/settings/
    """

    gpt_4: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    default_max_completion_tokens: int = 16000
    default_temperature: float = 0
    
    # Vertex AI settings
    vertex_enabled: bool = True 
    vertex_project_id: str = "website-382700"
    vertex_location: str = "us-central1"

    gemini_2: str = "gemini-2.0-flash-exp"


# Create an AgentSettings object
agent_settings = AgentSettings()
