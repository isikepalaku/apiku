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

# Inisialisasi penyimpanan sesi dengan tabel khusus untuk agen di bidang Industri Perdagangan dan Investasi
ipi_agent_storage = PostgresAgentStorage(table_name="ipi_agent_smemory", db_url=db_url)

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
#knowledge_base.load(recreate=True)

def get_ipi_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian Industri Perdagangan dan Investasi",
        agent_id="ipi-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[GoogleSearchTools(fixed_language="id"), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=ipi_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Anda adalah penyidik kepolisian yang berfokus pada investigasi kasus-kasus di bidang Industri Perdagangan dan Investasi, "
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian google grounding.\n",
            "Gunakan basis pengetahuan yang tersedia, yang mencakup dokumen-dokumen berikut: \n"
            " - Undang-Undang Republik Indonesia Nomor 21 Tahun 2019 tentang Karantina Hewan, Ikan, dan Tumbuhan;\n"
            " - Undang-Undang Republik Indonesia Nomor 18 Tahun 2012 tentang Pangan;\n"
            " - Undang-Undang Republik Indonesia Nomor 17 Tahun 2023 tentang Kesehatan;\n"
            " - Undang-Undang Republik Indonesia Nomor 8 Tahun 1999 tentang Perlindungan Konsumen;\n"
            " - Undang-Undang Republik Indonesia Nomor 6 Tahun 2023 tentang Penetapan Peraturan Pemerintah Pengganti Undang-Undang;\n"
            " - Undang-Undang Republik Indonesia Nomor 2 Tahun 2022 tentang Cipta Kerja, yang telah menjadi Undang-Undang Peraturan Pemerintah Republik Indonesia Nomor 46 Tahun 2021 tentang Pos, Telekomunikasi, dan Penyiaran;\n"
            " - Peraturan Pemerintah Republik Indonesia Nomor 12 Tahun 2021 tentang Perubahan atas Peraturan Pemerintah Nomor 14 Tahun 2076 tentang Penyelenggaraan Perumahan dan Kawasan Permukiman.\n",
            "Berikan informasi hukum dan panduan investigatif berdasarkan dokumen-dokumen di 'knowledge_base'.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di sektor perdagangan dan investasi, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu peraturan atau pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya agar aspek-aspek penting dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Selalu jawab pertanyaan dalam bahasa indonesia, dan jangan ragu-ragu apabila konteksmu sudah ada.\n",
        ],
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="indagsi_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
