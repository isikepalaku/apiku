import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from custom_tools.googlescholar import GoogleScholarTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.postgres import PostgresStorage
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage
memory = Memory(db=PostgresMemoryDb(table_name="ite_agent_memories", db_url=db_url))
ite_agent_storage = PostgresStorage(table_name="ite_agent_memory", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU ITE
knowledge_base = TextKnowledgeBase(
    path=Path("data/ite"),  # Pastikan folder ini berisi dokumen-dokumen UU ITE
    vector_db=PgVector(
        table_name="text_ite",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_ite_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian (Ahli UU ITE)",
        agent_id="ite-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[GoogleScholarTools(), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=ite_agent_storage,
        search_knowledge=True,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        memory=memory,
        description=(
            "Anda adalah penyidik kepolisian spesialisasi (UU) Nomor 1 Tahun 2024 Perubahan Kedua atas Undang-Undang Nomor 11 Tahun 2008 tentang ITE."
        ),
        instructions=[
            "Berikan informasi hukum dan panduan investigatif berdasarkan knowledge base yang disediakan\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di dunia digital, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Anjurkan untuk berkonsultasi dengan penyidik atau ahli hukum resmi apabila situasi hukum tertentu memerlukan analisis atau penanganan lebih lanjut.\n",
        ],
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )
