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
memory = Memory(db=PostgresMemoryDb(table_name="narkotika_agent_memories", db_url=db_url))
narkotika_agent_storage = PostgresAgentStorage(table_name="narkotika_agent_memory", db_url=db_url, auto_upgrade_schema=True)
COLLECTION_NAME = "narkotika"

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU Narkotika
knowledge_base = TextKnowledgeBase(
    path=Path("data/NARKOTIKA"),  # Pastikan folder ini berisi dokumen-dokumen terkait hukum dan regulasi perdagangan serta investasi
    vector_db=PgVector(
        table_name="text_narkotika",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_narkotika_agent(
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
        name="Penyidik Senior Tindak Pidana Narkotika",
        agent_id="narkotika-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash", temperature=0.2),
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(), 
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=narkotika_agent_storage,
        search_knowledge=True,
        description=(
            "Anda adalah asisten penyidik senior kepolisian spesialisasi Tindak Pidana Narkotika yang berperan sebagai mentor."
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian TavilyTools.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana narkotika.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Kamu adalah mentor penyidik senior yang sangat ahli dalam penanganan kasus narkotika dan memberikan panduan yang detail tentang analisis kasus serta aspek hukumnya.\n",
            "Knowledge base mu dibekali UU Nomor 35 Tahun 2009 tentang Narkotika dengan memperhatikan perubahan berdasarkan:\n",
            "- UU No. 1 Tahun 2023 (KUHP) yang mencabut sebagian (Pasal 111-126)\n",
            "- UU No. 6 Tahun 2023 tentang Cipta Kerja\n",
            "Berikan penjelasan yang komprehensif dengan mempertimbangkan status terkini dari peraturan tersebut."
            "Berikan panduan investigatif yang jelas dan terstruktur dalam bahasa indonesia\n",
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
