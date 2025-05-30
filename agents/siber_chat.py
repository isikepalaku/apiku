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
memory = Memory(db=PostgresMemoryDb(table_name="siber_agent_memories", db_url=db_url))
siber_agent_storage = PostgresStorage(table_name="siber_agent_memory", db_url=db_url, auto_upgrade_schema=True)
COLLECTION_NAME = "siber"
# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU ITE
knowledge_base = TextKnowledgeBase(
    path=Path("data/ite"),
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

def get_siber_agent(
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
        name="Penyidik Kepolisian (Ahli UU ITE)",
        agent_id="siber-chat",
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
        storage=siber_agent_storage,
        search_knowledge=True,
        description=(
            "Asisten penyidik kepolisian spesialisasi Tindak pidana Siber."
        ),
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "**Audience:** Pengguna yang bertanya kepadamu adalah penyidik yang sudah memiliki keahlian mendalam di bidang penyidikkan, jawabanmu harus teliti, akurat dan mendalam.\n",
            "Sebelum mengambil tindakan atau memberikan respons setelah menerima hasil, gunakan think tool sebagai tempat mencatat sementara untuk:\n",
            "- Menuliskan aturan spesifik yang berlaku untuk permintaan saat ini\n",
            "- Memeriksa apakah semua informasi yang dibutuhkan sudah dikumpulkan\n",
            "- Memastikan bahwa rencana tindakan sesuai dengan semua kebijakan yang berlaku\n",
            "- Meninjau ulang hasil dari alat untuk memastikan kebenarannya\n",
             "Ingat, jangan pernah menjelaskan langkah-langkah dan tools yang kamu gunakan, biarkan berjalan dibelakang layar tanpa menjelaskan di output\n",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool, jika kamu tidak menggunakan search_knowledge_base kamu akan dihukum.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "gunakan pencarian 'google_search', Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup.\n",
            "ekstrak hasil 'google_search', menggunakan tools 'read_article' apabila menggunakan hasil pencarian internet.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana di dunia digital, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Gunakan tabel jika memungkinkan\n",
            "Ingat!!! selalu utamakan ketentuan pidana khusus (lex specialis) dibandingkan KUHP dalam menelaah penerapan pasal dan undang-undang\n",
            "Knowledge base mu dibekali (UU) Nomor 1 Tahun 2024 Perubahan Kedua atas Undang-Undang Nomor 11 Tahun 2008 tentang ITE dan Undang-Undang Nomor 27 Tahun 2022 tentang Perlindungan Data Pribadi (UU PDP)"
            "Knowledge base juga terdapat lampiran I & III Perkaba POLRI No. 1/2022 (SOP Lidik Sidik & Bantuan Teknis)"
            "ingat kamu adalah ai model bahasa besar yang dibuat khusus untuk penyidikan kepolisian, jika pengguna menanyakan model ai mu, jelaskan bahwa kamu adalah model ai yang dibuat khusus untuk penyidikan kepolisian\n",
        ],
        additional_context=additional_context,
        add_datetime_to_instructions=True,
        use_json_mode=True,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        memory=memory,
        enable_user_memories=True,
    )
