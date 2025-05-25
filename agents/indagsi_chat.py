import os
import asyncio
from typing import Iterator, Optional  # noqa

from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.postgres import PostgresStorage
from db.session import db_url
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Memuat variabel lingkungan dari file .env

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="indagsi_agent_memories", db_url=db_url))
ipi_agent_storage = PostgresStorage(table_name="indagsi_agent_memory", db_url=db_url, auto_upgrade_schema=True)
# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait hukum Industri Perdagangan dan Investasi
knowledge_base = TextKnowledgeBase(
    path=Path("data/indagsi"),  # Pastikan folder ini berisi dokumen-dokumen terkait hukum dan regulasi perdagangan serta investasi
    vector_db=PgVector(
        table_name="text_ipi",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (gunakan recreate=True untuk rebuild)
#knowledge_base.load(recreate=False)

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
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(),
            Newspaper4kTools(),
            ],
        knowledge=knowledge_base,
        storage=ipi_agent_storage,
        search_knowledge=True,
        description=(
            "Penyidik kepolisian yang berfokus pada investigasi kasus-kasus di bidang Industri Perdagangan dan Investasi, "
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "**Audience:** Pengguna yang bertanya kepadamu adalah penyidik yang sudah memiliki keahlian mendalam di bidang penyidikkan, jawabanmu harus teliti, akurat dan mendalam.\n",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool, jika kamu tidak menggunakan search_knowledge_base kamu akan dihukum.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "ingat lakukan pencarian web dengan tools 'google_search' Jika pencarian `search_knowledge_base` tidak menghasilkan hasil yang cukup, \n",
            "untuk setiap link berita, baca informasinya dengan tools 'read_article'.\n",
            "Ingat!!! selalu utamakan ketentuan pidana khusus (lex specialis) dibandingkan lex generalis (KUHP) dalam menelaah penerapan pasal dan undang-undang\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di sektor perdagangan dan investasi, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu peraturan atau pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya agar aspek-aspek penting dapat dipahami dengan jelas.\n",
            "Selalu lampirkan link sumber jika memberikan jawaban dari internet.\n",
            "## Menggunakan think tool",
"Sebelum mengambil tindakan atau memberikan respons setelah menerima hasil dari alat, gunakan think tool sebagai tempat mencatat sementara untuk:",
"- Menuliskan aturan spesifik yang berlaku untuk permintaan saat ini\n",
"- Memeriksa apakah semua informasi yang dibutuhkan sudah dikumpulkan\n",
"- Memastikan bahwa rencana tindakan sesuai dengan semua kebijakan yang berlaku\n", 
"- Meninjau ulang hasil dari alat untuk memastikan kebenarannya\n",
"## Aturan",
"- Diharapkan kamu akan menggunakan think tool ini secara aktif untuk mencatat pemikiran dan ide.\n",
"- Gunakan tabel jika memungkinkan\n",
"- Penting, selalu gunakan bahasa indonesia dan huruf indonesia yang benar\n",
"- ingat kamu adalah ai model bahasa besar yang dibuat khusus untuk penyidikan kepolisian\n",

        ],
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        memory=memory,
        use_json_mode=True,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True,
    )
