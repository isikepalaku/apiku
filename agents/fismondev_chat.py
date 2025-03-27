import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.models.google import Gemini
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools

load_dotenv()  # Load environment variables from .env file

# Initialize storage
fismondev_agent_storage = PostgresAgentStorage(table_name="fismondev_agent_memory", db_url=db_url)
COLLECTION_NAME = "fismondev"

# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/p2sk"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=GeminiEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)
# Load knowledge base before initializing agent
#knowledge_base.load(recreate=False)

def get_fismondev_agent(
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
        name="fismondev Chat",
        agent_id="fismondev-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[GoogleSearchTools(fixed_language="id"), Newspaper4kTools()],
        knowledge=knowledge_base,
        storage=fismondev_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description="Anda adalah penyidik kepolisian Fismodev (Fiskal moneter dan devisa).",
        instructions=[
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Jika pencarian basis pengetahuan tidak menghasilkan hasil yang cukup, gunakan pencarian google grounding.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana dalam sektor jasa keuangan, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Anjurkan untuk berkonsultasi dengan penyidik atau ahli hukum resmi apabila situasi hukum tertentu memerlukan analisis atau penanganan lebih lanjut.\n",
            "Gunakan hasil pencarian web jika tidak ditemukan jawaban di knowledge base mu.\n",
            """
Catatan: KETENTUAN PIDANA DALAM UU FIDUSIA
## Pasal 35 uu fidusia
Setiap orang yang dengan sengaja memalsukan, mengubah, menghilangkan atau dengan cara apapun memberikan keterangan secara menyesatkan,  
yang  jika  hal  tersebut  diketahui  oleh  salah  satu  pihak tidak  melahirkan  perjanjian  Jaminan  Fidusia,  dipidana  dengan  pidana penjara paling singkat 1 (satu) tahun dan paling lama 5 (lima) tahun 
dan denda  paling  sedikit  Rp.10.000.000,-(sepuluh  juta  rupiah)  dan  paling banyak Rp.100.000.000,- (seratus juta rupiah).

Pasal 36 uu fidusia
Pemberi Fidusia  yang  mengalihkan,  menggadaikan,  atau  menyewakan Benda  yang  menjadi  objek  Jaminan  Fidusia  sebagaimana  dimaksud dalam  Pasal  23  ayat  (2)  yang  dilakukan  tanpa  persetujuan  
tertulis terlebih dahulu dari Penerima Fidusia, dipidana dengan pidana penjara paling  lama  2  (dua)  tahun  dan  denda  paling  banyak  Rp.50.000.000,(lima puluh juta rupiah)."""
        ],
        additional_context=additional_context,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True
    )
