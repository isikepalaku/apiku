import os
from agno.media import File
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.models.deepseek import DeepSeek
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.storage.postgres import PostgresStorage
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType
from db.session import db_url
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="wassidik_chat_agent_memories", db_url=db_url))
wassidik_chat_agent_storage = PostgresStorage(table_name="wassidik_chat_agent_memory", db_url=db_url, auto_upgrade_schema=True)
COLLECTION_NAME = "wassidik"
# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait Wassidik
# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait hukum untuk Tipidter
knowledge_base = TextKnowledgeBase(
    path=Path("data/wassidik"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key=os.getenv("QDRANT_API_KEY"),
        search_type=SearchType.hybrid,
    )
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=False)

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
        model=Gemini(id="gemini-2.0-flash"),
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(cache_results=True), 
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=wassidik_chat_agent_storage,
        search_knowledge=True,
        description=(
            "Anda adalah Asisten Wassidik (Pengawasan Penyidikan) Polri yang bertugas untuk melakukan koordinasi dan pengawasan terhadap proses penyelidikan dan penyidikan tindak pidana. Fungsi ini dilaksanakan oleh unit seperti Bagwassidik di tingkat Polda atau Birowassidik di tingkat Mabes Polri."
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "Sebelum mengambil tindakan atau memberikan respons setelah menerima hasil, gunakan think tool sebagai tempat mencatat sementara untuk:\n",
            "- Menuliskan aturan spesifik yang berlaku untuk permintaan saat ini\n",
            "- Memeriksa apakah semua informasi yang dibutuhkan sudah dikumpulkan\n",
            "- Memastikan bahwa rencana tindakan sesuai dengan semua kebijakan yang berlaku\n",
            "- Meninjau ulang hasil dari alat untuk memastikan kebenarannya\n",
            "**Audience:** Pengguna yang bertanya kepadamu adalah penyidik yang yang ingin mengetahui aturan-aturan penyidikan, jawabanmu harus teliti, akurat dan mendalam.\n",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
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
        use_json_mode=True,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        memory=memory,
        enable_user_memories=True,
    )
