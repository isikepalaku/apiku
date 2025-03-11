import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.memory import AgentMemory
from agno.memory.db.postgres import PgMemoryDb
from db.session import db_url

load_dotenv()  # Load environment variables from .env file

# Initialize storage
perkaba_agent_storage = PostgresAgentStorage(table_name="baru.perkaba_agent_sessions", db_url=db_url)

# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/perkaba"),
    vector_db=PgVector(
        table_name="text_perkabaku",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=True, upsert=True)

def get_perkaba_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="SOP Reskrim Agent",
        agent_id="sop-reskrim-agent",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"), # Fixed model name
        knowledge=knowledge_base,
    # Add a tool to search the knowledge base which enables agentic RAG.
    # This is enabled by default when `knowledge` is provided to the Agent.
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        storage=perkaba_agent_storage,
        description="Anda agen AI yang dirancang untuk membantu dalam proses investigasi dan penegakan hukum berdasarkan Standar Operasional Prosedur (SOP) yang diatur dalam Peraturan Kepala Badan Reserse Kriminal Polri Nomor 1 Tahun 2022.",
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Cari informasi terkait penyusunan dan pelaksanaan administrasi penyelidikan serta penyidikan tindak pidana dalam basis pengetahuan SOP yang tersedia.",
            
            "Berikan panduan yang jelas dan rinci sesuai prosedur yang tercantum dalam dokumen SOP, khususnya yang berkaitan dengan penyusunan MINDIK (Administrasi Penyidikan) seperti Laporan Polisi, Berita Acara, Surat Perintah, dan dokumen terkait lainnya.",
            
            "Semua tindakan harus sesuai dengan pedoman SOP, termasuk penyusunan dokumen administrasi, pengelolaan barang bukti, proses wawancara, observasi, dan prosedur teknis lainnya.",
            "Perhatikan perbedaan tahap penyidikan dan penyelidikan, tahap penyelidikan belum mengharuskan upaya paksa kecuali dalam hal tertangkap tangan",
        ],
        memory=AgentMemory(
            db=PgMemoryDb(table_name="perkaba_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        debug_mode=debug_mode,
        add_datetime_to_instructions=True,
        show_tool_calls=False,
        markdown=True,
    )
