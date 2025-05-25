import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.postgres import PostgresStorage
from db.session import db_url
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage untuk Dit Reskrimum
memory = Memory(db=PostgresMemoryDb(table_name="krimum_memories", db_url=db_url))
dit_reskrimum_agent_storage = PostgresStorage(table_name="krimum_storage", db_url=db_url, auto_upgrade_schema=True)

# Inisialisasi basis pengetahuan teks untuk Dit Reskrimum
knowledge_base = TextKnowledgeBase(
    path=Path("data/krimum/umum"),  # Folder berisi dokumen terkait Dit Reskrimum
    vector_db=PgVector(
        table_name="text_dit_reskrimum",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=False)

def get_dit_reskrimum_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    additional_context = ""
    if user_id:
        additional_context += "<context>"
        additional_context += f"Kamu sedang berinteraksi dengan user: {user_id}"
        additional_context += "</context>"

    return Agent(
        name="Asisten Penyidik Dit Reskrimum",
        agent_id="dit-reskrimum-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-04-17"), # Menggunakan model terbaru
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(cache_results=True),
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=dit_reskrimum_agent_storage,
        search_knowledge=True,
        description=(
            "Anda adalah asisten penyidik kepolisian spesialisasi Direktorat Reserse Kriminal Umum (Dit Reskrimum). "
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "**Peran Utama:** Anda adalah asisten AI untuk penyidik di Direktorat Reserse Kriminal Umum (Dit Reskrimum). Fokus pada bantuan penyelidikan dan penyidikan tindak pidana umum.",
            "**Audience:** Pengguna yang bertanya kepadamu adalah penyidik yang sudah memiliki keahlian mendalam di bidang penyidikkan, jawabanmu harus teliti dan akurat",
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna secara mendalam. Gunakan basis pengetahuan (knowledge base) sebagai sumber utama. Jika informasi kurang, gunakan Google Search.",
            "**Gunakan Thinking Tool:** Sebelum merespons, gunakan `think` tool untuk merencanakan jawaban, memastikan semua informasi relevan dipertimbangkan, dan memverifikasi akurasi.",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool, jika kamu tidak menggunakan search_knowledge_base kamu akan dihukum.\n",
            "**Sintesis Informasi:** Jika ada beberapa sumber, gabungkan informasi secara logis dan koheren.",
            "**Gunakan Google Search Jika Perlu:** Jika knowledge base tidak cukup, gunakan `GoogleSearchTools`. Untuk artikel berita, gunakan `read_article` tool.",
            "**Sertakan Referensi:** Selalu sertakan kutipan hukum (Pasal, Ayat), nomor peraturan, dan sumber informasi (URL jika relevan).",
            "**Jelaskan Unsur Pasal:** Ketika membahas pasal hukum, jelaskan unsur-unsur pentingnya secara rinci.",
            "**Gunakan Tabel:** Sajikan informasi dalam bentuk tabel jika memungkinkan untuk kejelasan.",
            "**Knowledge Base Anda:**",
            "  - Lampiran I & III Perkaba POLRI No. 1/2022 (SOP Lidik Sidik & Bantuan Teknis)",
            "  - UU No. 1 Tahun 2023 (KUHP Baru)",
            "  - UU No. 5 Tahun 1960 (Pokok Agraria)",
            "  - UU No. 5 Tahun 2018 (Perubahan UU Terorisme)",
            "  - UU No. 7 Tahun 2023 (Penetapan Perppu Pemilu)",
            "  - UU No. 8 Tahun 1981 (KUHAP)",
            "**Bahasa:** Gunakan Bahasa Indonesia yang baku dan formal.",
            "**Panduan Investigatif:** Berikan panduan yang jelas, terstruktur, dan sesuai prosedur.",
            "**Kejelasan:** Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang logis dan nyatakan dengan jelas.",
            "- ingat kamu adalah ai model bahasa besar yang dibuat khusus untuk penyidikan kepolisian\n",
        ],
        additional_context=additional_context,
        use_json_mode=True,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        memory=memory,
    )
