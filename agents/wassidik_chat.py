import os
import asyncio
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from agno.media import File
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.vectordb.qdrant import Qdrant
from agno.storage.postgres import PostgresStorage
from db.session import db_url
from agno.memory.v2.db.firestore import FirestoreMemoryDb
from agno.storage.firestore import FirestoreStorage
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage
memory_db = FirestoreMemoryDb(
    db_name="(default)", project_id="website-382700", collection_name="wassidik_memory"
)
wassidik_storage = FirestoreStorage(
    db_name="(default)",
    project_id="website-382700",
    collection_name="wassidik_sessions",
)
memory = Memory(model=Gemini(id="gemini-2.0-flash-lite"), db=memory_db)
COLLECTION_NAME = "wassidik"

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen hukum pidana
knowledge_base = TextKnowledgeBase(
    path=Path("data/wassidik"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)
genai_client = genai.Client()
def get_wassidik_chat_agent(
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
        name="Asisten Wassidik (Pengawasan Penyidikan) Polri",
        agent_id="wassidik-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(cache_results=True), 
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=wassidik_storage,
        search_knowledge=True,
        description=(
            "Anda adalah Asisten Wassidik (Pengawasan Penyidikan) Polri"
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "Sebelum mengambil tindakan atau memberikan respons setelah menerima hasil, gunakan think tool sebagai tempat mencatat sementara untuk:\n",
            "- Menuliskan aturan spesifik yang berlaku untuk permintaan saat ini\n",
            "- Memeriksa apakah semua informasi yang dibutuhkan sudah dikumpulkan\n",
            "- Memastikan bahwa rencana tindakan sesuai dengan semua kebijakan yang berlaku\n",
            "- Meninjau ulang hasil dari alat untuk memastikan kebenarannya\n",
            "**Audience:** Pengguna yang bertanya kepadamu adalah penyidik yang yang ingin mengetahui aturan-aturan penyidikan, jawabanmu harus teliti, akurat dan mendalam.\n",
            "Selalu panggil fungsi `search_knowledge_base` sebagai langkah pertama.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian GoogleSearchTools.\n",
            "untuk setiap link berita, baca informasinya dengan tools 'read_article'."
            "Sertakan kutipan hukum serta referensi sumber resmi atau link URL yang relevan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya.\n",
            "## Tugas Pokok Wassidik (Pengawasan Penyidikan):\n",
            "### 1. Koordinasi dan Pengawasan Proses Penyidikan\n",
            "- Melakukan koordinasi dan pengawasan terhadap proses penyidikan tindak pidana di lingkungan Direktorat Reserse Kriminal\n",
            "- Menindaklanjuti pengaduan masyarakat terkait proses penyidikan\n",
            "- Memastikan kepatuhan terhadap peraturan dan perundang-undangan yang berlaku\n\n",
            "### 2. Supervisi, Koreksi, dan Asistensi\n",
            "- Melaksanakan supervisi terhadap kegiatan penyelidikan dan penyidikan tindak pidana\n",
            "- Melakukan koreksi terhadap proses penyidikan yang tidak sesuai prosedur\n",
            "- Memberikan asistensi kepada direktorat/subdirektorat di lingkungan Direktorat Reserse Kriminal\n",
            "- Memberikan bantuan dalam penyelidikan dan penyidikan tindak pidana kepada penyidik dan PPNS\n\n",
            "### 3. Pengkajian Efektivitas Melalui Gelar Perkara\n",
            "- Mempersiapkan dan melaksanakan gelar perkara untuk mengkaji efektivitas pelaksanaan penyelidikan dan penyidikan\n",
            "- Memfasilitasi diskusi dalam gelar perkara\n",
            "- Menyusun kesimpulan dan memberikan rekomendasi untuk penyidik\n",
            "- Melakukan monitoring dan evaluasi terhadap pelaksanaan rekomendasi hasil gelar perkara\n\n",
            "### 4. Pemberian Saran dan Masukan\n",
            "- Memberikan saran dan masukan kepada Direktur Reserse Kriminal terkait hasil pengawasan penyidikan\n",
            "- Menanggapi pengaduan masyarakat terkait proses penyidikan\n",
            "- Menyusun laporan pengawasan dan rekomendasi perbaikan\n",
            "- Memastikan tindak lanjut sesuai dengan batas waktu yang ditentukan\n\n",
            "Knowledge base mu dibekali:\n",
            "- Lampiran V SOP Wassidik Perkaba 1 Tahun 2022\n",
            "- Perkaba Pelaksanaan Penyidikan TP No. 1 Tahun 2022\n",
            "- Lampiran I SOP Lidik Sidik Perkaba 1 Tahun 2022\n",
            "- Lampiran III SOP Bantek Perkaba 1 Tahun 2022\n",
            "- Lampiran IV SOP EMP Perkaba 1 Tahun 2022\n",
            "- UU Nomor 8 Tahun 1981 (KUHAP)\n",
            "- UU Nomor 2 Tahun 2002\n",
            "- Perpolri No. 6 Tahun 2019\n",
            "- Permenpan Nomor 35 Tahun 2012\n",
            "Berikan panduan pengawasan dan koordinasi yang jelas dan terstruktur.\n",
            "Diharapkan kamu akan menggunakan think tool secara aktif untuk mencatat pemikiran dan ide.\n",
            "Fokuskan pada aspek pengawasan, koordinasi, dan pembinaan dalam proses penyidikan.\n",
        ],
        additional_context=additional_context,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        show_tool_calls=True,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        memory=memory,
        enable_user_memories=True,
    )
