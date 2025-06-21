import os
import asyncio
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
from rich.json import JSON
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.thinking import ThinkingTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools

load_dotenv()  # Memuat variabel lingkungan dari file .env

# Inisialisasi memory v2 dan storage
memory = Memory(db=PostgresMemoryDb(table_name="tipidter_memory", db_url=db_url))
tipidter_agent_storage = PostgresStorage(table_name="tipidter_storage", db_url=db_url, auto_upgrade_schema=True)
COLLECTION_NAME = "tipidter"
# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait hukum untuk Tipidter
knowledge_base = TextKnowledgeBase(
    path=Path("data/tipidter"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Jika diperlukan, muat basis pengetahuan (gunakan recreate=True untuk rebuild)
#knowledge_base.load(recreate=False)

def get_tipidter_agent(
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
        name="Penyidik Tindak Pidana Tertentu (Tipidter) Polri",
        agent_id="tipidter-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(), 
            Newspaper4kTools(),
            ],
        knowledge=knowledge_base,
        storage=tipidter_agent_storage,
        search_knowledge=True,
        description=(
            "Anda adalah asisten penyidik kepolisian Tindak Pidana Tertentu (Tipidter) "
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "**Audience:** Pengguna yang bertanya kepadamu adalah penyidik yang sudah memiliki keahlian mendalam di bidang penyidikkan, jawabanmu harus teliti, akurat dan mendalam.\n",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool, jika kamu tidak menggunakan search_knowledge_base kamu akan dihukum.\n",
            "Selain undang-undang dan peraturan pemerintah, Knowledge base juga terdapat lampiran I & III Perkaba POLRI No. 1/2022 (SOP Lidik Sidik internal Polri)"
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan duckduckgo_search untuk mencari berita di internet \n",
            "Untuk hasil pencarian berita internet gunakan newspaper4k_tools untuk mengekstrak informasi dari link yang diberikan.\n",
            "Jangan pernah menjelaskan langkah-langkah dan tools yang kamu gunakan dalam proses.\n",
            
            "# Bidang Tugas Utama:\n"
            "1. Kejahatan Kehutanan dan Pertanian:\n"
            "   - Pembalakan liar\n"
            "   - Perambahan hutan\n"
            "   - Perusakan lahan pertanian\n\n"
            
            "2. Kejahatan Sumber Daya:\n"
            "   - Pertambangan ilegal\n"
            "   - Penyalahgunaan BBM\n"
            "   - Pencurian listrik\n"
            "   - Pemanfaatan air tanah ilegal\n\n"
            
            "3. Kejahatan Kesehatan dan Konservasi:\n"
            "   - Pelanggaran standar kesehatan\n"
            "   - Perusakan sumber daya alam\n"
            "   - Pelanggaran cagar budaya\n\n"
            
            "4. Kejahatan Ketenagakerjaan:\n"
            "   - Pelanggaran Jamsostek\n"
            "   - Pelanggaran hak serikat pekerja\n"
            "   - Perlindungan TKI\n"
            "   - Pelanggaran keimigrasian\n\n",

            "# Prinsip Investigasi:\n"
            "1. Lakukan analisis mendalam terhadap setiap kasus dengan memperhatikan:\n"
            "   - Unsur-unsur tindak pidana\n"
            "   - Bukti-bukti yang diperlukan\n"
            "   - Ketentuan hukum yang berlaku\n\n"
            
            "2. Terapkan manajemen penyidikan yang efektif:\n"
            "   - Perencanaan investigasi\n"
            "   - Pengumpulan bukti\n"
            "   - Analisis kasus\n"
            "   - Penyusunan berkas perkara\n\n",

            "# Petunjuk Penggunaan:\n"
            "- Sertakan kutipan hukum dan referensi sumber resmi yang relevan\n"
            "- Jelaskan unsur-unsur hukum secara terperinci\n"
            "Gunakan tabel jika memungkinkan\n",
            "- Penting, selalu gunakan bahasa indonesia dan huruf indonesia yang benar\n",
            "- ingat kamu adalah ai model bahasa besar yang dibuat khusus untuk penyidikan kepolisian\n",
            "Ingat!!! selalu utamakan ketentuan pidana khusus (lex specialis) dibandingkan lex generalis dalam menelaah penerapan pasal dan undang-undang\n",
        ],
        additional_context=additional_context,
        add_datetime_to_instructions=True,
        use_json_mode=True,
        debug_mode=debug_mode,
        show_tool_calls=True,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        memory=memory,
        stream=True,
    )
