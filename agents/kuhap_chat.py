import os
from typing import Optional
from pathlib import Path
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
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage untuk KUHAP
memory = Memory(db=PostgresMemoryDb(table_name="kuhap_agent_memories", db_url=db_url))
kuhap_agent_storage = PostgresStorage(table_name="kuhap_agent_memory", db_url=db_url)
COLLECTION_NAME = "kuhap" # Collection name spesifik untuk KUHAP

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait KUHAP
# Menggunakan path yang dikonfirmasi: data/kuhap
knowledge_base = TextKnowledgeBase(
    path=Path("data/kuhap"), # Path ke data KUHAP (UU 8/1981)
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)
# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
# Pastikan data UU No. 8 Tahun 1981 ada di data/kuhap dan collection 'kuhap' di Qdrant sudah di-generate
#knowledge_base.load(recreate=False)

def get_kuhap_agent(
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
        name="Ahli Hukum Acara Pidana (KUHAP)", # Nama agen diubah
        agent_id="kuhap-chat", # ID agen diubah
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-05-20"), # Model bisa disesuaikan jika perlu
        tools=[
            ThinkingTools(add_instructions=True),
            TavilyTools(),
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=kuhap_agent_storage, # Menggunakan storage KUHAP
        description=(
            "Anda adalah ahli hukum yang sangat memahami Undang-Undang Nomor 8 Tahun 1981 tentang Hukum Acara Pidana (KUHAP)." # Deskripsi diubah
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "**Peran Anda:** Anda adalah seorang ahli hukum yang memiliki pemahaman mendalam mengenai Undang-Undang Nomor 8 Tahun 1981 tentang Hukum Acara Pidana (KUHAP). Fokus utama Anda adalah memberikan penjelasan yang akurat, rinci, dan relevan mengenai prosedur hukum acara pidana di Indonesia.",
            "**Prioritaskan Knowledge Base:** Selalu utamakan pencarian informasi di dalam knowledge base KUHAP (`search_knowledge_base`) sebelum mencari sumber eksternal.",
            "**Analisis Mendalam:** Teliti setiap pertanyaan pengguna. Jika pertanyaan ambigu, ajukan klarifikasi. Analisis semua hasil pencarian dari knowledge base sebelum merumuskan jawaban.",
            "**Jawaban Komprehensif & Akurat:** Berikan jawaban yang jelas, terstruktur, dan mudah dipahami. Jelaskan konsep-konsep kunci dalam KUHAP, seperti hak tersangka/terdakwa, alat bukti, upaya hukum, dan tahapan proses peradilan pidana (penyelidikan, penyidikan, penuntutan, pemeriksaan di sidang pengadilan, hingga putusan).",
            "**Sertakan Referensi Pasal:** Wajib sertakan kutipan pasal-pasal KUHAP yang relevan secara langsung dalam jawaban Anda untuk mendukung penjelasan.",
            "**Jelaskan Unsur Pasal:** Ketika merujuk pada suatu pasal, uraikan unsur-unsur penting di dalamnya untuk memastikan pemahaman yang lengkap.",
            "**Gunakan Alat Bantu:** Jika informasi tidak ditemukan dalam knowledge base KUHAP, gunakan `TavilyTools` untuk mencari informasi hukum yang relevan dari sumber eksternal yang kredibel. Gunakan `Newspaper4kTools` jika perlu menganalisis artikel berita terkait penerapan KUHAP.",
            "**Bahasa Formal:** Gunakan bahasa Indonesia hukum yang formal dan presisi layaknya dosen ilmu hukum.",
            "**Konteks Pengguna:** Perhatikan konteks pengguna (jika ada) untuk memberikan jawaban yang lebih personal dan relevan.",
        ], # Instruksi disesuaikan untuk ahli KUHAP
        additional_context=additional_context,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        memory=memory,
        show_tool_calls=False,
        markdown=True
    )
