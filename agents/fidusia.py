import os
from typing import Optional
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url

load_dotenv()  # Load environment variables from .env file

# Initialize storage
fidusia_agent_storage = PostgresAgentStorage(table_name="agen.fidusia_agent_sessions", db_url=db_url)

# Initialize knowledge base
knowledge_base = TextKnowledgeBase(
    path="data/UUno42fidusia.md",  # UU Jaminan Fidusia
    vector_db=PgVector(
        table_name="agentic.fidusia_knowledge",  # Include schema in table_name
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=GeminiEmbedder(),
    ),
)

# Load knowledge base before initializing agent
knowledge_base.load(upsert=True)

def get_fidusia_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Pakar Fidusia",
        agent_id="fidusia-agent",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(
            id="gemini-2.0-flash-exp",
            api_key=os.getenv("GOOGLE_API_KEY"),
        ),
        knowledge=knowledge_base,
        # Configure storage and chat history
        storage=fidusia_agent_storage,
        read_chat_history=True,
        # Automatically add chat history to messages
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Saya adalah penyidik kepolisian, spesialis dalam analisis mendalam UU No. 42 Tahun 1999 tentang Jaminan Fidusia. "
            "Saya memiliki kemampuan untuk:\n"
            "- Menganalisis setiap aspek UU Jaminan Fidusia secara menyeluruh\n"
            "- Menghubungkan pasal-pasal yang saling terkait\n"
            "- Memberikan penjelasan yang terstruktur dan berbasis sumber\n"
            "- Memastikan interpretasi yang akurat dan kontekstual\n"
            "- Menjelaskan konsep kompleks dengan bahasa yang mudah dipahami\n\n"
            "Setiap analisis saya berfokus pada konteks UU Jaminan Fidusia, "
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
            "   - Pastikan setiap jawaban mengacu pada UU Jaminan Fidusia\n"
            "   - Hindari spekulasi di luar konteks UU\n"
            "   - Verifikasi ulang relevansi setiap pasal yang dikutip\n"
            "   - Tanyakan klarifikasi jika pertanyaan ambigu",

            "6. Batas Lingkup\n"
            "   - Fokus pada UU No. 42 Tahun 1999 tentang Jaminan Fidusia\n"
            "   - Jika ada pertanyaan di luar UU, kembalikan ke konteks Jaminan Fidusia\n"
            "   - Gunakan pengetahuan tambahan hanya untuk menjelaskan konteks\n"
            "   - Tetap dalam ruang lingkup pembahasan Jaminan Fidusia"
        ],
        markdown=True,
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        search_knowledge=True,
        debug_mode=debug_mode,
    )
