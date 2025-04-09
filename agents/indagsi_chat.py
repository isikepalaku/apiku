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
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.tools.reasoning import ReasoningTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

load_dotenv()  # Memuat variabel lingkungan dari file .env

# Inisialisasi penyimpanan sesi dengan tabel khusus untuk agen di bidang Industri Perdagangan dan Investasi
ipi_agent_storage = PostgresAgentStorage(table_name="indagsi_agent_memory", db_url=db_url)
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
        use_json_mode=True,
        tools=[
            ReasoningTools(),
            GoogleSearchTools(), 
            Newspaper4kTools(),
            ],
        knowledge=knowledge_base,
        storage=ipi_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Penyidik kepolisian yang berfokus pada investigasi kasus-kasus di bidang Industri Perdagangan dan Investasi, "
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan 'search_knowledge_base' tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "ingat lakukan pencarian web dengan tools 'google_search' Jika pencarian `search_knowledge_base` tidak menghasilkan hasil yang cukup, \n",
            "untuk setiap link berita, baca informasinya dengan tools 'read_article'.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di sektor perdagangan dan investasi, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu peraturan atau pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya agar aspek-aspek penting dapat dipahami dengan jelas.\n",
            "Selalu lampirkan link sumber jika memberikan jawaban dari internet.\n",
        ],
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )
