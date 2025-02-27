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

load_dotenv()  # Load environment variables from .env file

# Inisialisasi penyimpanan sesi dengan tabel baru khusus untuk agen UU Cipta Kerja
cipta_kerja_agent_storage = PostgresAgentStorage(table_name="cipta_kerja_agent_sessions", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU Cipta Kerja
knowledge_base = TextKnowledgeBase(
    path=Path("data/ciptakerja"),  # Pastikan folder ini berisi dokumen-dokumen UU Cipta Kerja
    vector_db=PgVector(
        table_name="text_cipta_kerja",
        db_url=db_url,
        embedder=GeminiEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_cipta_kerja_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian (Ahli UU Cipta Kerja)",
        agent_id="cipta-kerja-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-exp", grounding=True),
        knowledge=knowledge_base,
        storage=cipta_kerja_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Saya adalah penyidik kepolisian yang memiliki spesialisasi dalam Undang-Undang Cipta Kerja. "
        ),
        instructions=[
            "Catatan: Undang-Undang Cipta Kerja telah diubah dengan:\n"
            "- UU No. 66 Tahun 2024 tentang Perubahan Ketiga atas Undang-Undang Nomor 17 Tahun 2008 tentang Pelayaran\n"
            "- UU No. 65 Tahun 2024 tentang Perubahan Ketiga atas Undang-Undang Nomor 13 Tahun 2016 tentang Paten\n"
            "- UU No. 63 Tahun 2024 tentang Perubahan Ketiga atas Undang-Undang Nomor 6 Tahun 2011 tentang Keimigrasian\n"
            "- UU No. 3 Tahun 2024 tentang Perubahan Kedua atas Undang-Undang Nomor 6 Tahun 2014 tentang Desa\n",
            "Berikan informasi hukum dan panduan investigatif berdasarkan knowledge base yang disediakan, khususnya terkait dengan UU Cipta Kerja.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek yang diatur dalam UU Cipta Kerja, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Anjurkan untuk berkonsultasi dengan penyidik atau ahli hukum resmi apabila situasi hukum tertentu memerlukan analisis atau penanganan lebih lanjut.\n",
        ],
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="cipta_kerja_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
