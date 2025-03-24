import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent, AgentMemory
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory.db.postgres import PgMemoryDb

load_dotenv()  # Load environment variables from .env file

# Inisialisasi penyimpanan sesi dengan tabel baru khusus untuk agen ITE
siber_agent_storage = PostgresAgentStorage(table_name="siber_agent_memory", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU ITE
knowledge_base = TextKnowledgeBase(
    path=Path("data/ite"),  # Pastikan folder ini berisi dokumen-dokumen UU ITE
    vector_db=PgVector(
        table_name="text_siber11",
        db_url=db_url,
        embedder=GeminiEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_siber_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian (Ahli UU ITE)",
        agent_id="siber-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[TavilyTools(), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=siber_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Anda adalah penyidik kepolisian spesialisasi Tindak pidana Siber."
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian google grounding.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di dunia digital, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Knowledge base mu dibekali (UU) Nomor 1 Tahun 2024 Perubahan Kedua atas Undang-Undang Nomor 11 Tahun 2008 tentang ITE dan Undang-Undang Nomor 27 Tahun 2022 tentang Perlindungan Data Pribadi (UU PDP)"
        ],
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="siber_agent_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
