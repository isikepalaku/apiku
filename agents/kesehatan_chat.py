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
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage
memory = Memory(db=PostgresMemoryDb(table_name="kesehatan_agent_memories", db_url=db_url))
kesehatan_agent_storage = PostgresStorage(table_name="kesehatan_agent_memory", db_url=db_url)

# Initialize knowledge base with health law documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/kesehatan"),  # Pastikan folder ini berisi dokumen-dokumen terkait UU Kesehatan
    vector_db=PgVector(
        table_name="text_uu_kesehatan",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_kesehatan_agent(
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
        name="Penyidik Kepolisian (Ahli UU Kesehatan)",
        agent_id="uu-kesehatan-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-001"),
        tools=[
            GoogleSearchTools(), 
            Newspaper4kTools(),
            MCPTools(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
                )
            )
        ],
        knowledge=knowledge_base,
        storage=kesehatan_agent_storage,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        search_knowledge=True,
        memory=memory,
        enable_user_memories=True,
        enable_session_summaries=True,
        description=(
            "Anda adalah penyidik kepolisian yang memiliki spesialisasi dalam "
            "Undang-Undang Republik Indonesia Nomor 17 Tahun 2023 tentang Kesehatan. "
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian google grounding.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek yang diatur dalam UU Republik Indonesia Nomor 17 Tahun 2023 tentang Kesehatan, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Anjurkan untuk berkonsultasi dengan penyidik atau ahli hukum resmi apabila situasi hukum tertentu memerlukan analisis atau penanganan lebih lanjut.\n",
        ],
        additional_context=additional_context,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )
