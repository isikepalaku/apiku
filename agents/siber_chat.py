import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.vectordb.qdrant import Qdrant
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="siber_agent_memories", db_url=db_url))
siber_agent_storage = PostgresAgentStorage(table_name="siber_agent_memory", db_url=db_url, auto_upgrade_schema=True)
COLLECTION_NAME = "siber"
# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU ITE
knowledge_base = TextKnowledgeBase(
    path=Path("data/ite"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=GeminiEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_siber_agent(
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
        name="Penyidik Kepolisian (Ahli UU ITE)",
        agent_id="siber-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash", temperature=0.2),
        tools=[
            ThinkingTools(),
            GoogleSearchTools(), 
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=siber_agent_storage,
        search_knowledge=True,
        description=(
            "Anda adalah asisten penyidik kepolisian spesialisasi Tindak pidana Siber."
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "gunakan pencarian 'google_search', Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup.\n",
            "ekstrak hasil 'google_search', menggunakan tools 'read_article' apabila menggunakan hasil pencarian internet.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di dunia digital, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Kamu adalah asisten penyidik kepolisian sehingga buat dirimu layaknya mentor anggota kepolisian yang sangat ahli tindaak pidana siber.\n",
            "Gunakan tabel jika memungkinkan\n",
            "Knowledge base mu dibekali (UU) Nomor 1 Tahun 2024 Perubahan Kedua atas Undang-Undang Nomor 11 Tahun 2008 tentang ITE dan Undang-Undang Nomor 27 Tahun 2022 tentang Perlindungan Data Pribadi (UU PDP)"
            "- Penting, selalu gunakan bahasa indonesia dan huruf indonesia yang benar\n",
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
