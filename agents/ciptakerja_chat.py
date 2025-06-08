import os
from typing import Optional
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
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi penyimpanan sesi dengan tabel baru khusus untuk agen UU Cipta Kerja
cipta_kerja_agent_storage = PostgresStorage(table_name="cipta_agen_memory", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU Cipta Kerja
knowledge_base = TextKnowledgeBase(
    path=Path("data/ciptakerja"),  # Pastikan folder ini berisi dokumen-dokumen UU Cipta Kerja
    vector_db=PgVector(
        table_name="text_cipta_kerja",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
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
        model=Gemini(id="gemini-2.5-flash-preview-05-20", vertexai=True),
        tools=[ThinkingTools(add_instructions=True), GoogleSearchTools(fixed_language="id"), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=cipta_kerja_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Anda adalah ahli Undang-undang Nomor 6 Tahun 2023 tentang Cipta Kerja. "
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "Catatan: Undang-Undang Cipta Kerja telah diubah dengan:\n"
            "- UU No. 66 Tahun 2024 tentang Perubahan Ketiga atas Undang-Undang Nomor 17 Tahun 2008 tentang Pelayaran\n"
            "- UU No. 65 Tahun 2024 tentang Perubahan Ketiga atas Undang-Undang Nomor 13 Tahun 2016 tentang Paten\n"
            "- UU No. 63 Tahun 2024 tentang Perubahan Ketiga atas Undang-Undang Nomor 6 Tahun 2011 tentang Keimigrasian\n"
            "- UU No. 3 Tahun 2024 tentang Perubahan Kedua atas Undang-Undang Nomor 6 Tahun 2014 tentang Desa\n",
            "Berikan informasi hukum dan panduan investigatif berdasarkan knowledge base yang disediakan, khususnya terkait dengan UU Cipta Kerja.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek yang diatur dalam UU Cipta Kerja, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Jangan pernah menjelaskan langkah-langkah yang kamu lakukan, gunakan tools dan knowledgebase tanpa menjelaskan prosesnya.\n",
        ],
        use_json_mode=True,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )
