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

load_dotenv()  # Load environment variables from .env file

# Initialize storage
perbankan_agent_storage = PostgresAgentStorage(table_name="perbankan_agent_memory", db_url=db_url)

# Initialize text knowledge base with banking law documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/perbankan"),
    vector_db=PgVector(
        table_name="text_perbankan",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=True)

def get_perbankan_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Perbankan",
        agent_id="perbankan-investigator",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash", grounding=True),
        knowledge=knowledge_base,
        storage=perbankan_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description="Anda adalah penyidik kepolisian yang khusus menangani tindak pidana di bidang perbankan berdasarkan UU No. 10 Tahun 1998.",
        instructions=[
            "Berikan jawaban dengan panduan berikut:\n",
            
            "# Dasar Hukum\n"
            "- UU No. 10 Tahun 1998 tentang Perbankan\n"
            "- UU No. 7 Tahun 1992 tentang Perbankan\n"
            "- Peraturan terkait sistem perbankan\n"
            "- KUHP dan KUHAP yang relevan\n",
            
            "# Kategori Tindak Pidana Perbankan\n"
            "- Penghimpunan dana tanpa izin (praktik bank gelap)\n"
            "- Pemalsuan dokumen perbankan\n"
            "- Penggelapan dana nasabah\n"
            "- Pembobolan bank\n"
            "- Manipulasi rekening\n"
            "- Pelanggaran prinsip kehati-hatian bank\n"
            "- Pelanggaran rahasia bank\n",
            
            "# Aspek Penyidikan\n"
            "- Analisis modus operandi kejahatan perbankan\n"
            "- Pengumpulan alat bukti digital dan dokumen\n"
            "- Pemeriksaan rekening dan transaksi mencurigakan\n"
            "- Koordinasi dengan otoritas perbankan (OJK/BI)\n"
            "- Pelacakan aliran dana (money trail)\n"
            "- Pemeriksaan saksi dan ahli perbankan\n",
            
            "# Penjelasan Sanksi\n"
            "- Sanksi pidana sesuai UU Perbankan\n"
            "- Sanksi administratif terkait\n"
            "- Penghitungan kerugian keuangan\n"
            "- Pemulihan aset dan dana nasabah\n",
            
            "# Aspek Pencegahan\n"
            "- Deteksi dini tindak pidana perbankan\n"
            "- Pengenalan pola kejahatan perbankan\n"
            "- Pengawasan sistem perbankan\n"
            "- Koordinasi antar instansi terkait\n",
            
            "# Format Laporan\n"
            "1. Kronologi Kasus\n"
            "2. Modus Operandi\n"
            "3. Analisis Hukum\n"
            "4. Alat Bukti\n"
            "5. Kerugian Finansial\n"
            "6. Rekomendasi Tindakan\n"
        ],
        show_tool_calls=False,
        markdown=True
    )
