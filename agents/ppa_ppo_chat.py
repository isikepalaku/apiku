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
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="ppa_ppo_agent_memories", db_url=db_url))
ppa_ppo_agent_storage = PostgresAgentStorage(table_name="ppa_ppo_agent_memory", db_url=db_url, auto_upgrade_schema=True)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU PPA PPO
knowledge_base = TextKnowledgeBase(
    path=Path("data/krimum/tpo"),  # Folder berisi dokumen UU terkait PPA dan PPO
    vector_db=PgVector(
        table_name="text_ppa_ppo",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=False)

def get_ppa_ppo_agent(
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
        name="Penyidik Senior Tindak Pidana Perempuan dan Anak serta Pidana Perdagangan Orang",
        agent_id="ppa-ppo-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(), 
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=ppa_ppo_agent_storage,
        search_knowledge=True,
        description=(
            "Anda adalah asisten penyidik kepolisian spesialisasi Tindak Pidana Perempuan dan Anak serta Pidana Perdagangan Orang (Dir PPA PPO) yang berperan sebagai mentor."
        ),
        instructions=[
            "Kamu adalah mentor penyidik senior yang sangat ahli dalam penanganan kasus kekerasan terhadap perempuan dan anak serta perdagangan orang.\n",
            "Sebelum mengambil tindakan atau memberikan respons setelah menerima hasil, gunakan think tool sebagai tempat mencatat sementara untuk:\n",
            "- Menuliskan aturan spesifik yang berlaku untuk permintaan saat ini\n",
            "- Memeriksa apakah semua informasi yang dibutuhkan sudah dikumpulkan\n",
            "- Memastikan bahwa rencana tindakan sesuai dengan semua kebijakan yang berlaku\n",
            "- Meninjau ulang hasil dari alat untuk memastikan kebenarannya\n",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian GoogleSearchTools.\n",
            "untuk setiap link berita, baca informasinya dengan tools 'read_article'."
            "Sertakan kutipan hukum serta referensi sumber resmi atau link URL yang relevan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya.\n",
            "Gunakan tabel jika memungkinkan.\n",
            "Knowledge base mu dibekali:\n",
            "- UU Nomor 23 Tahun 2004 tentang Penghapusan Kekerasan Dalam Rumah Tangga (PKDRT)\n",
            "- UU Nomor 21 Tahun 2007 tentang Pemberantasan Tindak Pidana Perdagangan Orang (TPPO)\n",
            "  Perhatikan perubahan berdasarkan UU No. 1 Tahun 2023 (KUHP) yang mencabut sebagian Pasal 2\n",
            "- UU Nomor 11 Tahun 2012 tentang Sistem Peradilan Pidana Anak (SPPA)\n", 
            "- UU Nomor 12 Tahun 2022 tentang Tindak Pidana Kekerasan Seksual (TPKS)\n",
            "Penting, selalu gunakan bahasa indonesia dan huruf indonesia yang benar.\n",
            "Berikan panduan investigatif yang jelas dan terstruktur.\n",
            "Diharapkan kamu akan menggunakan think tool secara aktif untuk mencatat pemikiran dan ide.\n",
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
        enable_user_memories=True,
        enable_session_summaries=True,
    )
