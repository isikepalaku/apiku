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
from db.session import db_url

load_dotenv()  # Load environment variables from .env file

# Initialize storage
perkaba_agent_storage = PostgresAgentStorage(table_name="baru.perkaba_agent_sessions", db_url=db_url)

# Initialize text knowledge base with multiple documents
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://celebesbot.com/pdf/LAMPIRANISOPLIDIKSIDIKPERKABA1THN2022TGL27DES2022.pdf", "https://celebesbot.com/pdf/PERKABAPELAKSPENYIDIKANTPNO1TH2022TGL27DES2022.pdf"],
    vector_db=PgVector(
        table_name="text_perkabaku",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(upsert=True)

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
        model=OpenAIChat(id="gpt-4o-mini"),
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
            "Cari informasi terkait penyusunan dan pelaksanaan administrasi penyelidikan serta penyidikan tindak pidana dalam basis pengetahuan SOP yang tersedia.",
            
            "Berikan panduan yang jelas dan rinci sesuai prosedur yang tercantum dalam dokumen SOP, khususnya yang berkaitan dengan penyusunan MINDIK (Administrasi Penyidikan) seperti Laporan Polisi, Berita Acara, Surat Perintah, dan dokumen terkait lainnya.",
            
            "Semua tindakan harus sesuai dengan pedoman SOP, termasuk penyusunan dokumen administrasi, pengelolaan barang bukti, proses wawancara, observasi, dan prosedur teknis lainnya.",
        ],
        debug_mode=debug_mode,
        show_tool_calls=True,
        markdown=True,
    )
