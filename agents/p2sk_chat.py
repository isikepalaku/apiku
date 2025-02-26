import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory import AgentMemory
from agno.memory.db.postgres import PgMemoryDb

load_dotenv()  # Load environment variables from .env file

# Inisialisasi penyimpanan sesi dengan tabel baru khusus untuk agen P2SK
p2sk_agent_storage = PostgresAgentStorage(table_name="p2sk_agent_sessions", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU P2SK
knowledge_base = TextKnowledgeBase(
    path=Path("data/p2sk"),  # Pastikan folder ini berisi dokumen-dokumen UU P2SK
    vector_db=PgVector(
        table_name="text_p2sk",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
knowledge_base.load(recreate=True)

def get_p2sk_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian (Ahli UU P2SK)",
        agent_id="p2sk-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-exp", search=True),
        knowledge=knowledge_base,
        storage=p2sk_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=("anda adalah seorang penyidik kepolisian yang ahli UU P2SK"
                    "Berdasarkan PP nomor 5 tahun 2023 tentang Penyidikan tindak pidana di sektor jasa keuangan"
        ),
        instructions=[
            "Berikan jawaban dengan panduan berikut:\n",
            "# Dasar Hukum\n"
            "- Selalu mengacu pada UU dan peraturan yang relevan\n"
            "- Menyebutkan pasal-pasal spesifik yang terkait\n"
            "- Menjelaskan interpretasi hukum secara jelas\n",
            
            "# Kategori Pelanggaran\n"
            "- Menjelaskan jenis-jenis pelanggaran di sektor keuangan\n"
            "- Menguraikan unsur-unsur pidana yang harus terpenuhi\n"
            "- Memberikan contoh kasus yang relevan\n",
            
            "# Penjelasan Sanksi\n"
            "- Menerangkan sanksi pidana yang dapat diterapkan sesuai undang-undang\n"
            "- Menjelaskan sanksi tambahan jika ada, termasuk denda dan hukuman penjara\n"
            "- Membahas precedent hukum terkait\n",
            
            "# Referensi\n"
            "- Menyertakan sumber hukum yang dirujuk\n"
            "- Mengutip yurisprudensi relevan\n",
        ],
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="p2sk_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
