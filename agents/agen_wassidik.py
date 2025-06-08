import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.storage.postgres import PostgresStorage
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage
memory = Memory(db=PostgresMemoryDb(table_name="wassidik_agent_memories", db_url=db_url))
wassidik_agent_storage = PostgresStorage(table_name="wassidik_agent_memory", db_url=db_url)
COLLECTION_NAME = "perkabapolri"
# Initialize text knowledge base with multiple documents
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://celebesbot.com/pdf/LAMPIRANVSOPWASSIDIKPERKABA1TH2022TGL27DES2022.pdf", "https://celebesbot.com/pdf/PERKABAPELAKSPENYIDIKANTPNO1TH2022TGL27DES2022.pdf"],
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=False)

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
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        memory=memory,
        enable_user_memories=True,
        enable_session_summaries=True,
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
