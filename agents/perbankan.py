import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.models.google import Gemini
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.postgres import PostgresStorage
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url
# from agno.tools.mcp import MCPTools
# from mcp import StdioServerParameters
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage
memory = Memory(db=PostgresMemoryDb(table_name="perbankan_agent_memories", db_url=db_url))
perbankan_agent_storage = PostgresStorage(table_name="perbankan_agent_memory", db_url=db_url)

# Initialize text knowledge base with banking law documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/perbankan"),
    vector_db=PgVector(
        table_name="text_perbankan",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(),
    ),
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=True)

def get_perbankan_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Perbankan",
        agent_id="perbankan-investigator",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[
            TavilyTools(), 
            Newspaper4kTools(),
            # MCPTools(
            #     server_params=StdioServerParameters(
            #         command="npx",
            #         args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
            #     )
            # )
        ],
        knowledge=knowledge_base,
        storage=perbankan_agent_storage,
        search_knowledge=True,
        memory=memory,
        enable_user_memories=True,
        enable_session_summaries=True,
        description="Anda adalah penyidik kepolisian yang ahli menjelaskan tindak pidana di bidang perbankan berdasarkan UU No. 10 Tahun 1998.",
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian web.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek yang diatur dalam UU No. 10 Tahun 1998 tentang perbankan, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Anjurkan untuk berkonsultasi dengan penyidik atau ahli hukum resmi apabila situasi hukum tertentu memerlukan analisis atau penanganan lebih lanjut.\n",
        ],
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )
