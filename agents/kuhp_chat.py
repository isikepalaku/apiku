import asyncio
import os
from typing import Optional
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.qdrant import Qdrant
from agno.storage.postgres import PostgresStorage
from db.session import db_url
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools
# from agno.tools.mcp import MCPTools
# from mcp import StdioServerParameters

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="kuhp_agent_memories", db_url=db_url))
kuhp_agent_storage = PostgresStorage(table_name="kuhp_agent_memory", db_url=db_url)
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
    # Create MCPTools instance separately
    # sequential_thinking_mcp_tools = MCPTools(
    #     server_params=StdioServerParameters(
    #         command="npx",
    #         args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
    #     )
    # )

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
        model=Gemini(id="gemini-2.5-flash-preview-04-17", vertexai=True),
        tools=[
            TavilyTools(),
            Newspaper4kTools(),
            # sequential_thinking_mcp_tools, # Use the created instance
        ],
        knowledge=knowledge_base,
        storage=kuhp_agent_storage,
        description=(
            "Ahli UU Nomor 1 Tahun 2023 tentang KUHP"
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "Ingat selalu awali dengan pencarian di knowledge_base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana, "
            "ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek "
            "penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Gunakan TavilyTools apabila pertanyaan tidak ditemukan di knowledge_base.\n",
            "**Sequential Thinking:**",
            "Sebelum mengambil tindakan atau merespons pengguna setelah menerima hasil alat, gunakan alat `think` sebagai coretan untuk:",
            "- Mencantumkan aturan spesifik yang berlaku untuk permintaan saat ini",
            "- Memeriksa apakah semua informasi yang diperlukan telah dikumpulkan",
            "- Memverifikasi bahwa tindakan yang direncanakan mematuhi semua kebijakan",
            "- Mengulangi hasil alat untuk kebenaran",
            "**Aturan Sequential Thinking:**",
            "- Diharapkan Anda akan menggunakan alat `think` secara bebas untuk mencatat pemikiran dan ide.",
            "- Gunakan tabel jika memungkinkan.",
        ],
        additional_context=additional_context,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        memory=memory,
        show_tool_calls=False,
        markdown=True
    )
