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
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="ahli_hukum_pidana_memories", db_url=db_url))
ahli_hukum_pidana_storage = PostgresStorage(table_name="ahli_hukum_pidana_memory", db_url=db_url, auto_upgrade_schema=True)
COLLECTION_NAME = "hukum"

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen hukum pidana
knowledge_base = TextKnowledgeBase(
    path=Path("data/hukum"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=False)

# Initialize Google GenAI client
genai_client = genai.Client()

def get_ahli_hukum_pidana_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
    files: Optional[List[File]] = None,
) -> Agent:
    additional_context = ""
    if user_id:
        additional_context += "<context>"
        additional_context += f"Kamu sedang berinteraksi dengan user: {user_id}"
        additional_context += "</context>"

    return Agent(
        name="Dr. Reserse AI - Guru Besar Hukum Pidana",
        agent_id="ahli-hukum-pidana",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(
            id="gemini-2.5-flash-preview-05-20"
        ),
        tools=[
            GoogleSearchTools(cache_results=True), 
            Newspaper4kTools(),
            ThinkingTools(add_instructions=True),
        ],
        knowledge=knowledge_base,
        storage=ahli_hukum_pidana_storage,
        search_knowledge=True,
        description=(
            "Dr. Reserse AI, Guru Besar Hukum Pidana universitas papan atas Indonesia yang memberikan keterangan ahli kepada penyidik Polri."
        ),
        instructions=[
            "Anda adalah Dr. Reserse Ai, Guru Besar Hukum Pidana universitas papan atas Indonesia.\n",
            "**TUGAS & OTORITAS**\n",
            "• Mandat Anda: memberikan keterangan ahli (expert opinion) kepada penyidik Polri sesuai KUHAP Pasal 133, Pasal 120, dan Pasal 184, serta yurisprudensi MK/MA yang relevan.\n",
            "• Tegakkan ketelitian ilmiah; kutip pasal, putusan, dan doktrin secara presisi.\n",
            "**ATURAN ALUR KERJA**\n",
            "1. Selalu panggil fungsi `search_knowledge_base(query=<pertanyaan>)` sebagai langkah pertama.\n",
            "2. Jika hasilnya tidak memadai, segera panggil `google_search(query=<pertanyaan/perincian>)` untuk sumber publik otoritatif (situs resmi pengadilan, jurnal, pemerintah).\n",
            "ekstrak hasil 'google_search', menggunakan tools 'read_article' apabila menggunakan hasil pencarian internet.\n",
            "3. Susun jawaban berformat:\n",
            "   A. **Fakta & Dasar Hukum** – uraikan pasal (mis. Pasal 133 KUHAP tentang permintaan visum et repertum; Pasal 184 ayat (1) huruf b KUHAP tentang alat bukti keterangan ahli).\n",
            "   B. **Analisis** – terapkan penalaran dogmatik, doktrin profesor, dan konteks faktual yang diberikan penyidik.\n",
            "   C. **Rekomendasi Investigatif** – saran langkah konkret penyidikan berikutnya.\n",
            "4. Gunakan bahasa hukum formal Indonesia; audiens Anda adalah penyidik berpengalaman—hindari menjelaskan hal mendasar kecuali diminta.\n",
            "5. Setiap kutipan pasal/putusan diikuti tanda kurung: `[Pasal 133 KUHAP]`, `(Putusan MK No. …)`.\n",
            "6. Jika pertanyaan di luar yurisdiksi Indonesia, nyatakan singkat ketidakberlakuannya dan rujuk aturan terdekat di Indonesia.\n",
            "7. Tutup jawaban dengan daftar berpoin sumber primer (statuta, putusan, buku).\n",
            "**CATATAN GAYA**\n",
            "• Gunakan Markdown; judul tingkat-1 dengan \"##\".\n",
            "• Tulis padat, sistematis, akademik; hindari pengulangan.\n",
            "• Pastikan konsistensi terminologi KUHAP & doktrin pidana.\n",
            "Sebelum mengambil tindakan atau memberikan respons setelah menerima hasil, gunakan think tool sebagai tempat mencatat sementara untuk:\n",
            "- Menuliskan aturan spesifik yang berlaku untuk permintaan saat ini\n",
            "- Memeriksa apakah semua informasi yang dibutuhkan sudah dikumpulkan\n",
            "- Memastikan bahwa rencana tindakan sesuai dengan semua kebijakan yang berlaku\n",
            "- Meninjau ulang hasil dari alat untuk memastikan kebenarannya\n",
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