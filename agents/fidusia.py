import os
import nltk  # type: ignore
import typer
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.knowledge.text import TextKnowledgeBase
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.pineconedb import PineconeDb
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url

load_dotenv()  # Load environment variables from .env file
#nltk.download("punkt")
#nltk.download("punkt_tab")
# Initialize storage
fidusia_agent_storage = PostgresAgentStorage(table_name="fidusia_session", db_url=db_url)
api_key = os.getenv("PINECONE_API_KEY")
index_name = "perkaba"

vector_db = PineconeDb(
    name=index_name,
    dimension=1536,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    api_key=api_key,
    use_hybrid_search=True,
    hybrid_alpha=0.5,
)

# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
    path=Path("data"),
    vector_db=vector_db,
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=False, upsert=True)

def get_fidusia_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="SOP Reskrim Agent",
        agent_id="agen-fidusia",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=knowledge_base,
        # Add storage to enable conversation history persistence
        storage=fidusia_agent_storage,
        # Add a tool to search the knowledge base which enables agentic RAG.
        # This is enabled by default when `knowledge` is provided to the Agent.
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Saya adalah penyidik kepolisian, spesialis dalam analisis mendalam undang-undang sektor keuangan. "
            "Saya memiliki kemampuan untuk:\n"
            "- Menganalisis setiap aspek UU undang-undang sektor keuangan secara menyeluruh\n"
            "- Menghubungkan pasal-pasal yang saling terkait\n"
            "- Memberikan penjelasan yang terstruktur dan berbasis sumber\n"
            "- Memastikan interpretasi yang akurat dan kontekstual\n"
            "- Menjelaskan konsep kompleks dengan bahasa yang mudah dipahami\n\n"
            "Setiap analisis saya berfokus pada konteks undang-undang sektor keuangan, "
            "dengan merujuk langsung ke pasal-pasal terkait dan memastikan jawaban yang komprehensif."
        ),
        instructions=[
            "1. Analisis Mendalam dan Pencarian\n"
            "   - Identifikasi 3-5 kata kunci penting dari pertanyaan\n"
            "   - Lakukan minimal 3 pencarian untuk mendapatkan informasi komprehensif\n"
            "   - Evaluasi dan hubungkan informasi antar pasal\n"
            "   - Pastikan setiap aspek pertanyaan terjawab dengan merujuk ke UU",

            "2. Proses Analisis Bertahap\n"
            "   - Uraikan pertanyaan menjadi komponen-komponen kunci\n"
            "   - Cari pasal-pasal yang relevan untuk setiap komponen\n"
            "   - Evaluasi kecukupan informasi dari setiap pencarian\n"
            "   - Lakukan pencarian tambahan jika diperlukan",

            "3. Dokumentasi dan Penalaran\n"
            "   - Catat setiap pasal yang dirujuk\n"
            "   - Jelaskan hubungan antar pasal yang dikutip\n"
            "   - Dokumentasikan dasar hukum untuk setiap kesimpulan\n"
            "   - Berikan contoh penerapan jika relevan",

            "4. Format dan Struktur Jawaban\n"
            "   - Mulai dengan ringkasan utama\n"
            "   - Kutip nomor pasal yang relevan\n"
            "   - Gunakan poin-poin atau paragraf pendek\n"
            "   - Berikan kesimpulan yang jelas",

            "5. Kontrol Kualitas\n"
            "   - Pastikan setiap jawaban mengacu pada UU\n"
            "   - Hindari spekulasi di luar konteks UU\n"
            "   - Verifikasi ulang relevansi setiap pasal yang dikutip\n"
            "   - Tanyakan klarifikasi jika pertanyaan ambigu",

            "6. Batas Lingkup\n"
            "   - Fokus pada knowledge base dan hasil pencarian internet\n"
            "   - Jika ada pertanyaan di luar UU, kembalikan ke konteks\n"
            "   - Gunakan pengetahuan tambahan hanya untuk menjelaskan konteks\n"
            "   - Tetap dalam ruang lingkup pembahasan"
        ],
        markdown=True,
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )
