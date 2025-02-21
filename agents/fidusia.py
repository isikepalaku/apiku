import os
from typing import Optional
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.googlesearch import GoogleSearchTools
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from dotenv import load_dotenv

load_dotenv()

# Initialize storage for session management
fidusia_agent_storage = PostgresAgentStorage(table_name="fidusia_session", db_url=db_url)

knowledge_base = TextKnowledgeBase(
    path=Path("data"),
    # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
    vector_db=PgVector(
        table_name="uup2sk_data",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)
# Load the knowledge base: Comment after first run as the knowledge base is already loaded
#knowledge_base.load(upsert=True)

def get_web_agent(debug_mode: bool = False) -> Agent:
    return Agent(
        name="Penelusur Web Fidusia",
        role="Mencari dan menganalisis informasi fidusia dari web",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[GoogleSearchTools(fixed_language="id")],
        instructions=dedent("""\
            Anda adalah peneliti web yang ahli dalam analisis fidusia! 🔍\
        """),
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )

def get_knowledge_agent(debug_mode: bool = False) -> Agent:
    return Agent(
        name="Penganalisis Knowledge Base",
        role="Menganalisis dan mengintegrasikan informasi dari basis pengetahuan fidusia",
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=knowledge_base,
    # Add a tool to search the knowledge base which enables agentic RAG.
    # This is enabled by default when `knowledge` is provided to the Agent.
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

def get_fidusia_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    web_agent = get_web_agent(debug_mode)
    knowledge_agent = get_knowledge_agent(debug_mode)
    
    return Agent(
        name="Tim Analisis Fidusia",
        role="Menganalisis dan melaporkan informasi fidusia secara komprehensif",
        agent_id="agen-fidusia",
        user_id=user_id,
        session_id=session_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        storage=fidusia_agent_storage,
        team=[web_agent, knowledge_agent],
        instructions=dedent("""\
            Anda adalah ketua tim analisis fidusia! ⚖️

            Peran Anda:
            1. Koordinasi antara penelitian web dan basis pengetahuan
            2. Sintesis temuan menjadi laporan hukum yang komprehensif
            3. Pastikan cakupan menyeluruh aspek fidusia
            4. Lacak perkembangan regulasi dan praktik
            5. Identifikasi preseden dan interpretasi penting
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )
