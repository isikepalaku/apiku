import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.qdrant import Qdrant
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from rich.json import JSON
from agno.memory.db.postgres import PgMemoryDb
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

load_dotenv()  # Memuat variabel lingkungan dari file .env

# Inisialisasi penyimpanan sesi dengan tabel khusus untuk agen Tipidter
tipidter_agent_storage = PostgresAgentStorage(table_name="tipidter_memory_agent", db_url=db_url)
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
#knowledge_base.load(recreate=True)

def get_tipidter_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    team_session_id: Optional[str] = None,
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
        model=Gemini(id="gemini-2.0-flash"),
        tools=[
            GoogleSearchTools(), 
            Newspaper4kTools(),
            MCPTools(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
                )
            )
        ],
        knowledge=knowledge_base,
        storage=tipidter_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=5,
        description=(
            "Anda adalah penyidik kepolisian yang bekerja di unit Tindak Pidana Tertentu (Tipidter) "
            "di bawah Subdit Tipidter Ditreskrimsus Polda. Anda bertugas menangani kasus-kasus khusus "
            "seperti kejahatan kehutanan, pertambangan ilegal, kesehatan, ketenagakerjaan, dan lainnya."
        ),
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian GoogleSearchTools.\n",
            
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
            "- Berikan panduan investigatif yang jelas dan terstruktur\n",
        ],
        additional_context=additional_context,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )
