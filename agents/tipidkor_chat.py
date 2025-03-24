import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent, AgentMemory
from agno.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory.db.postgres import PgMemoryDb
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools

load_dotenv()  # Load environment variables from .env file

# Initialize storage
tipidkor_agent_storage = PostgresAgentStorage(table_name="tipidkor_agent_memory", db_url=db_url)

# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/tipidkor"),
    vector_db=PgVector(
        table_name="text_tipidkor",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=True)

def get_tipidkor_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="TIPIDKOR Chat",
        agent_id="tipidkor-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[TavilyTools(), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=tipidkor_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description="Anda adalah penyidik kepolisian bidang tindak pidana korupsi.",
        instructions=[
            "Berikan jawaban dengan panduan berikut:\n",

            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian google grounding.\n",
            
            "# Dasar Hukum\n"
            "- Selalu mengacu pada UU dan peraturan yang relevan\n"
            "- Menyebutkan pasal-pasal spesifik yang terkait\n"
            "- Menjelaskan interpretasi hukum secara jelas\n",
            
            "# Kategori Tindak Pidana\n"
            "- Menjelaskan jenis-jenis tindak pidana korupsi\n"
            "- Menguraikan unsur-unsur pidana yang harus terpenuhi\n"
            "- Memberikan contoh kasus yang relevan\n",
            
            "# Penjelasan Sanksi\n"
            "- Menerangkan sanksi pidana yang dapat diterapkan\n"
            "- Menjelaskan sanksi tambahan jika ada\n"
            "- Membahas precedent hukum terkait\n",
            
            "# Referensi\n"
            "- Menyertakan sumber hukum yang dirujuk\n"
            "- Mengutip yurisprudensi relevan\n",
        ],
        debug_mode=debug_mode,
        # Store the memories and summary in a database
        memory=AgentMemory(
        db=PgMemoryDb(table_name="tipidkor_agent_memory", db_url=db_url),
        create_user_memories=True,
        create_session_summary=True,
    ),
        show_tool_calls=False,
        markdown=True
    )
