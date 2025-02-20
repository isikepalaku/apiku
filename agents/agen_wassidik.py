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
wassidik_agent_storage = PostgresAgentStorage(table_name="baru.wassidik_agent_sessions", db_url=db_url)

# Initialize text knowledge base with multiple documents
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://celebesbot.com/pdf/LAMPIRANVSOPWASSIDIKPERKABA1TH2022TGL27DES2022.pdf", "https://celebesbot.com/pdf/PERKABAPELAKSPENYIDIKANTPNO1TH2022TGL27DES2022.pdf"],
    vector_db=PgVector(
        table_name="text_wassidik",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(upsert=True)

def get_wassidik_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Pengawasan Penyidik Agent",
        agent_id="wassidik-agent",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=knowledge_base,
        storage=wassidik_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description="Anda adalah agen AI yang ahli dalam pengawasan penyidik (WASSIDIK) berdasarkan Peraturan Kepala Badan Reserse Kriminal Polri Nomor 1 Tahun 2022. Anda memahami seluruh aspek pengawasan dan evaluasi kinerja penyidik dalam proses penyidikan.",
        instructions=[
            "Berikan panduan tentang prosedur pengawasan penyidik sesuai dengan SOP WASSIDIK yang berlaku.",
            "Jelaskan mekanisme evaluasi kinerja penyidik, termasuk indikator dan parameter penilaian yang digunakan.",
            "Bantu mengidentifikasi potensi penyimpangan dalam proses penyidikan dan cara pencegahannya.",
            "Berikan rekomendasi untuk peningkatan kualitas pengawasan dan kinerja penyidik sesuai standar yang ditetapkan.",
            "Jelaskan tata cara pelaporan dan dokumentasi hasil pengawasan penyidik sesuai ketentuan yang berlaku."
        ],
        debug_mode=debug_mode,
        show_tool_calls=True,
        markdown=True
    )