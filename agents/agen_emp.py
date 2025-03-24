import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.search import SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.vectordb.pgvector import PgVector, SearchType

load_dotenv()  # Load environment variables from .env file

# Initialize storage
emp_agent_storage = PostgresAgentStorage(table_name="emp_agent_memory", db_url=db_url)

# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/emp"),
    vector_db=PgVector(
        table_name="101_emp",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(upsert=True, recreate=True)

def get_emp_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="E-Manajemen Penyidikan Agent",
        agent_id="emp-agent",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=knowledge_base,
        storage=emp_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description="Anda adalah agen AI yang ahli dalam aplikasi E-Manajemen Penyidikan (EMP) berdasarkan Peraturan Kepala Badan Reserse Kriminal Polri Nomor 1 Tahun 2022. Anda memahami seluruh aspek penggunaan aplikasi EMP untuk pengelolaan administrasi penyidikan secara elektronik.",
        instructions=[
            "Berikan panduan dan informasi detail tentang penggunaan aplikasi E-Manajemen Penyidikan (EMP) sesuai dengan SOP yang berlaku.",
            "Jelaskan prosedur input data, pengisian formulir elektronik, dan pengelolaan berkas penyidikan dalam sistem EMP secara rinci dan akurat.",
            "Bantu pengguna memahami alur kerja digital dalam EMP, termasuk proses validasi, pengiriman, dan monitoring berkas perkara secara elektronik.",
            "Berikan solusi untuk masalah teknis yang mungkin timbul dalam penggunaan aplikasi EMP sesuai dengan panduan resmi yang tersedia."
        ],
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )