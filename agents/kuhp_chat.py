import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.vectordb.qdrant import Qdrant
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="kuhp_agent_memories", db_url=db_url))
kuhp_agent_storage = PostgresAgentStorage(table_name="kuhp_agent_memory", db_url=db_url)
COLLECTION_NAME = "kuhp"
# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait KUHP
knowledge_base = TextKnowledgeBase(
    path=Path("data/kuhp"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)
# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=False)

def get_kuhp_agent(
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
        name="Penyidik Kepolisian (Ahli UU KUHP)",
        agent_id="kuhp-chat",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[
            ThinkingTools(add_instructions=True),
            TavilyTools(), 
            Newspaper4kTools(),
        ],
        storage=kuhp_agent_storage,
        description=(
            "Anda adalah penyidik kepolisian yang ahli dalam UU Nomor 1 Tahun 2023 tentang KUHP"
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge_base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana, "
            "ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek "
            "penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Gunakan TavilyTools apabila pertanyaan tidak ditemukan di knowledge_base.\n",
        ],
        additional_context=additional_context,
        debug_mode=debug_mode,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        memory=memory,
        enable_user_memories=True,
        enable_session_summaries=True,
        show_tool_calls=False,
        markdown=True
    )
