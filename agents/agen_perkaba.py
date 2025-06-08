import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.storage.postgres import PostgresStorage
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url
from agno.vectordb.qdrant import Qdrant
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage with fixed table names (as per Agno documentation)
memory = Memory(db=PostgresMemoryDb(table_name="perkaba_agent_memories", db_url=db_url))
perkaba_agent_storage = PostgresStorage(table_name="perkaba_agent_memory", db_url=db_url)
COLLECTION_NAME = "perkabapolri"

# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
path=Path("data/perkaba"),
vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=False)

def get_perkaba_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    
    return Agent(
        name="SOP Reskrim Agent",
        agent_id="sop-reskrim-agent",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"), # Fixed model name
        tools=[
            ThinkingTools(add_instructions=True),
        ],
        knowledge=knowledge_base,
        storage=perkaba_agent_storage,
        search_knowledge=True,
        description="Anda asisten penyidik polri yang membantu menjelaskan SOP penyelidikan dan penyidikan berdasarkan Peraturan Kepala Badan Reserse Kriminal Polri Nomor 1 Tahun 2022.",
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Cari informasi terkait penyusunan dan pelaksanaan administrasi penyelidikan serta penyidikan tindak pidana dalam basis pengetahuan SOP yang tersedia.",
            "Berikan panduan yang jelas dan rinci sesuai prosedur yang tercantum dalam dokumen SOP, khususnya yang berkaitan dengan penyusunan MINDIK (Administrasi Penyidikan) seperti Laporan Polisi, Berita Acara, Surat Perintah, dan dokumen terkait lainnya.",
            "Semua tindakan harus sesuai dengan pedoman SOP, termasuk penyusunan dokumen administrasi, pengelolaan barang bukti, proses wawancara, observasi, dan prosedur teknis lainnya.",
            "Perhatikan perbedaan tahap penyidikan dan penyelidikan, tahap penyelidikan belum mengharuskan upaya paksa kecuali dalam hal tertangkap tangan",
        ],
        memory=memory,
        enable_user_memories=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        enable_session_summaries=True,
        debug_mode=debug_mode,
        add_datetime_to_instructions=True,
        show_tool_calls=False,
        markdown=True,
    )
