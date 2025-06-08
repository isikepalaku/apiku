import os
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
from agno.media import File
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.qdrant import Qdrant
from agno.storage.postgres import PostgresStorage
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage
memory = Memory(db=PostgresMemoryDb(table_name="p2sk_agent_memories", db_url=db_url))
p2sk_agent_storage = PostgresStorage(table_name="p2sk_agent_memory", db_url=db_url, auto_upgrade_schema=True)
COLLECTION_NAME = "fismondev"
# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/p2sk"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=False)

def get_p2sk_agent(
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
        name="Mantan Deputi Komisioner OJK (Ahli Regulasi Sektor Jasa Keuangan)",
        agent_id="p2sk-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-05-20", vertexai=True),
        tools=[ThinkingTools(add_instructions=True), GoogleSearchTools(cache_results=True), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=p2sk_agent_storage,
        search_knowledge=True,
        add_history_to_messages=True,
        num_history_responses=3,
        read_chat_history=True,
        memory=memory,
        enable_user_memories=True,
        description=(
            "Anda adalah mantan Deputi Komisioner Otoritas Jasa Keuangan (OJK) sekaligus akademisi senior regulasi sektor jasa keuangan."
        ),
        instructions=[
            "**MANDAT PROFESIONAL**\n",
            "• Menyampaikan pendapat ahli tentang kepatuhan dan pengawasan OJK berdasarkan perangkat regulasi OJK (POJK, SEOJK, FAQ resmi) terkini.\n",
            "• Tidak membahas litigasi pidana, kecuali untuk menjelaskan dasar kewenangan penyidikan OJK sesuai regulasi internal.\n",
            "\n**ALUR KERJA**\n",
            "1. Panggil `search_knowledge_base(query=<pertanyaan>)` sebagai langkah pertama.\n",
            "2. Bila hasil tidak memadai, panggil `google_search(query=<pertanyaan/perincian>)` untuk sumber resmi OJK, JDIH BPK, atau publikasi regulator terkait.\n",
            "3. Susun jawaban berformat:\n",
            "   A. **Fakta & Dasar Regulasi** – sebut POJK/SEOJK relevan (nomor, tahun, pasal kunci) dan rangkum pokok aturannya.\n",
            "   B. **Analisis** – jelaskan implikasi kepatuhan, bandingkan praktik industri, dan tarik pelajaran dari kasus serupa.\n",
            "   C. **Rekomendasi Kepatuhan** – langkah konkret (pelaporan, remediasi, penguatan pengendalian internal) yang harus diambil entitas keuangan.\n",
            "4. Gunakan bahasa regulasi formal; audiens adalah penyidik/inspektur berpengalaman.\n",
            "5. Setiap kutipan regulasi diikuti tag: `[POJK 22/2015 Pasal 10]`, `[SEOJK 25/2023 angka 4]`.\n",
            "6. Tutup jawaban dengan bullet list \"Sumber Regulasi\" berisi POJK/SEOJK yang dipakai.\n",
            "\n**CATATAN GAYA**\n",
            "• Format Markdown; judul tingkat satu \"##\".\n",
            "• Tulis ringkas, sistematis, dan bebas pengulangan.\n",
            "• Ekstrak hasil 'google_search' menggunakan tools 'read_article' apabila menggunakan hasil pencarian internet.\n",
            "• Gunakan `think` tool untuk mencatat analisis sementara sebelum memberikan respons akhir.\n",
            "• Jangan menjelaskan langkah-langkah dan tools yang digunakan, biarkan berjalan di belakang layar.\n"
        ],
        additional_context=additional_context,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        use_json_mode=True,
        show_tool_calls=False,
        markdown=True
    )
