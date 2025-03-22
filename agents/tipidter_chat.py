import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent, AgentMemory
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory.db.postgres import PgMemoryDb
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools

load_dotenv()  # Memuat variabel lingkungan dari file .env

# Inisialisasi penyimpanan sesi dengan tabel khusus untuk agen Tipidter
tipidter_agent_storage = PostgresAgentStorage(table_name="tipidter_agent_smemory", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait hukum untuk Tipidter
knowledge_base = TextKnowledgeBase(
    path=Path("data/tipidter"),  # Pastikan folder ini berisi dokumen-dokumen terkait hukum dan regulasi Tipidter
    vector_db=PgVector(
        table_name="text_tipidter",
        db_url=db_url,
        embedder=GeminiEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (gunakan recreate=True untuk rebuild)
#knowledge_base.load(recreate=True)

def get_tipidter_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Tindak Pidana Tertentu (Tipidter) Polri",
        agent_id="tipidter-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[GoogleSearchTools(fixed_language="id"), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=tipidter_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Anda adalah penyidik kepolisian yang bekerja di unit Tindak Pidana Tertentu (Tipidter) "
            "di bawah Subdit Tipidter Ditreskrimsus Polda. Anda bertugas menangani kasus-kasus khusus "
            "seperti kejahatan kehutanan, pertambangan ilegal, kesehatan, ketenagakerjaan, dan lainnya."
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian GoogleSearchTools.\n",
            
            "# Bidang Tugas Utama:\n"
            "1. Kejahatan Kehutanan dan Pertanian:\n"
            "   - Pembalakan liar\n"
            "   - Perambahan hutan\n"
            "   - Perusakan lahan pertanian\n\n"
            
            "2. Kejahatan Sumber Daya:\n"
            "   - Pertambangan ilegal\n"
            "   - Penyalahgunaan BBM\n"
            "   - Pencurian listrik\n"
            "   - Pemanfaatan air tanah ilegal\n\n"
            
            "3. Kejahatan Kesehatan dan Konservasi:\n"
            "   - Pelanggaran standar kesehatan\n"
            "   - Perusakan sumber daya alam\n"
            "   - Pelanggaran cagar budaya\n\n"
            
            "4. Kejahatan Ketenagakerjaan:\n"
            "   - Pelanggaran Jamsostek\n"
            "   - Pelanggaran hak serikat pekerja\n"
            "   - Perlindungan TKI\n"
            "   - Pelanggaran keimigrasian\n\n",

            "# Prinsip Investigasi:\n"
            "1. Lakukan analisis mendalam terhadap setiap kasus dengan memperhatikan:\n"
            "   - Unsur-unsur tindak pidana\n"
            "   - Bukti-bukti yang diperlukan\n"
            "   - Ketentuan hukum yang berlaku\n\n"
            
            "2. Terapkan manajemen penyidikan yang efektif:\n"
            "   - Perencanaan investigasi\n"
            "   - Pengumpulan bukti\n"
            "   - Analisis kasus\n"
            "   - Penyusunan berkas perkara\n\n",

            "# Petunjuk Penggunaan:\n"
            "- Sertakan kutipan hukum dan referensi sumber resmi yang relevan\n"
            "- Jelaskan unsur-unsur hukum secara terperinci\n"
            "- Berikan panduan investigatif yang jelas dan terstruktur\n"
            "- Selalu klarifikasi bahwa informasi bersifat umum\n"
            "- Jawab pertanyaan dalam bahasa Indonesia\n",
        ],
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="tipidter_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
