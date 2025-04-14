import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage
memory = Memory(db=PostgresMemoryDb(table_name="bantek_agent_memories", db_url=db_url))
bantek_agent_storage = PostgresAgentStorage(table_name="bantek_agent_memory", db_url=db_url)

# Initialize text knowledge base with multiple documents
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://celebesbot.com/pdf/LAMPIRANIIISOPBANTEKPERKABA1THN202TGL27DES2022.pdf"],
    vector_db=PgVector(
        table_name="bantek_perkaba",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(upsert=True)

def get_perkaba_bantek_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Bantek Agent",
        agent_id="sop-bantek-agent",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=knowledge_base,
        search_knowledge=True,
        storage=bantek_agent_storage,
        memory=memory,
        enable_user_memories=True,
        enable_session_summaries=True,
        description="Anda adalah agen AI yang dirancang untuk memberikan penjelasan mengenai Standar Operasional Prosedur (SOP) bantuan teknis, sesuai dengan Peraturan Kepala Badan Reserse Kriminal Polri Nomor 1 Tahun 2022.",
        instructions=[
                    "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
                    "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
                    "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
                    "Jelaskan secara detail mekanisme dan prosedur bantuan teknis yang mendukung proses penyelidikan dan penyidikan tindak pidana, termasuk koordinasi antar unit, pengelolaan peralatan, dan penggunaan teknologi informasi.",
                    "Berikan panduan terstruktur mengenai SOP bantuan teknis, seperti panduan operasional peralatan, protokol komunikasi, dan prosedur dokumentasi digital.",
                    "Pastikan penjelasan Anda selaras dengan pedoman SOP, standar operasional yang ada di dalam knowledge base mu.",
                    "Fokuskan penjelasan pada implementasi praktis dan penerapan prosedur teknis yang mendukung efektivitas investigasi, sambil memastikan kesesuaian dengan peraturan yang berlaku."],
        debug_mode=debug_mode,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        show_tool_calls=False,
        markdown=True,
    )
