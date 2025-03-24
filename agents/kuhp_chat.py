import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent, AgentMemory
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.vectordb.pgvector import PgVector
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory.db.postgres import PgMemoryDb
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi penyimpanan sesi dengan tabel baru khusus untuk agen KUHP
kuhp_agent_storage = PostgresAgentStorage(table_name="kuhp_agent_memory", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait KUHP
knowledge_base = TextKnowledgeBase(
    path=Path("data/kuhp"),  # Pastikan folder ini berisi dokumen-dokumen KUHP
    vector_db=PgVector(
        table_name="text_kuhp",
        db_url=db_url,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_kuhp_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian (Ahli UU KUHP)",
        agent_id="kuhp-chat",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[TavilyTools(), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=kuhp_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Anda adalah penyidik kepolisian yang ahli dalam UU Nomor 1 Tahun 2023 tentang KUHP"
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana, "
            "ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek "
            "penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan prosedur resmi kepolisian.\n",
        ],
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="kuhp_agent_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
