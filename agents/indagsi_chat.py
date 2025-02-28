import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory import AgentMemory
from agno.memory.db.postgres import PgMemoryDb

load_dotenv()  # Memuat variabel lingkungan dari file .env

# Inisialisasi penyimpanan sesi dengan tabel khusus untuk agen di bidang Industri Perdagangan dan Investasi
ipi_agent_storage = PostgresAgentStorage(table_name="ipi_agent_sessions", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait hukum Industri Perdagangan dan Investasi
knowledge_base = TextKnowledgeBase(
    path=Path("data/indagsi"),  # Pastikan folder ini berisi dokumen-dokumen terkait hukum dan regulasi perdagangan serta investasi
    vector_db=PgVector(
        table_name="text_ipi",
        db_url=db_url,
        embedder=GeminiEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (gunakan recreate=True untuk rebuild)
knowledge_base.load(recreate=True)

def get_ipi_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian (Spesialis Industri Perdagangan dan Investasi)",
        agent_id="ipi-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-exp", grounding=True),
        knowledge=knowledge_base,
        storage=ipi_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Saya adalah penyidik kepolisian yang berfokus pada investigasi kasus-kasus di bidang Industri Perdagangan dan Investasi, "
            "berdasarkan regulasi dan hukum yang berlaku."
        ),
        instructions=[
            "Berikan informasi hukum dan panduan investigatif berdasarkan dokumen-dokumen terkait hukum Industri Perdagangan dan Investasi.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di sektor perdagangan dan investasi, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu peraturan atau pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya agar aspek-aspek penting dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Anjurkan untuk berkonsultasi dengan penyidik atau ahli hukum resmi apabila situasi hukum tertentu memerlukan analisis atau penanganan lebih lanjut.\n",
        ],
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="ipi_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
