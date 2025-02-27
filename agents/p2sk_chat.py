import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.google import GeminiEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.models.google import Gemini
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.memory import AgentMemory
from agno.memory.db.postgres import PgMemoryDb

load_dotenv()  # Load environment variables from .env file

# Inisialisasi penyimpanan sesi dengan tabel baru khusus untuk agen P2SK
p2sk_agent_storage = PostgresAgentStorage(table_name="p2sk_agent_sessions", db_url=db_url)

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait UU P2SK
knowledge_base = TextKnowledgeBase(
    path=Path("data/p2sk"),  # Pastikan folder ini berisi dokumen-dokumen UU P2SK
    vector_db=PgVector(
        table_name="text_p2sk",
        db_url=db_url,
        embedder=GeminiEmbedder(),
    ),
)

# Jika diperlukan, muat basis pengetahuan (dengan recreate=True jika ingin rebuild)
#knowledge_base.load(recreate=True)

def get_p2sk_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Penyidik Kepolisian (Ahli UU P2SK)",
        agent_id="p2sk-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-exp"),
        knowledge=knowledge_base,
        storage=p2sk_agent_storage,
        search_knowledge=True,
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        description=(
            "Saya adalah penyidik kepolisian Fismondev yang memiliki spesialisasi dalam penyidikan di sektor jasa keuangan."
        ),
        instructions=[
            "Ingat selalu berikan informasi hukum dan panduan investigatif berdasarkan knowledge base yang tersedia.\n",
            "Sertakan kutipan hukum serta referensi sumber resmi yang relevan, terutama terkait aspek-aspek penyidikan tindak pidana dalam sektor jasa keuangan, ketika menjawab pertanyaan.\n",
            "Ketika menjawab mengenai suatu pasal, jelaskan secara terperinci unsur-unsur hukum yang mendasarinya, sehingga aspek-aspek penting dalam pasal tersebut dapat dipahami dengan jelas.\n",
            "Selalu klarifikasi bahwa informasi yang diberikan bersifat umum dan tidak menggantikan nasihat hukum profesional ataupun prosedur resmi kepolisian.\n",
            "Anjurkan untuk berkonsultasi dengan penyidik atau ahli hukum resmi apabila situasi hukum tertentu memerlukan analisis atau penanganan lebih lanjut.\n",
            "Jangan pernah menjawab diluar knowledge base yang kamu miliki.\n",
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
        debug_mode=debug_mode,
        memory=AgentMemory(
            db=PgMemoryDb(table_name="p2sk_memory", db_url=db_url),
            create_user_memories=True,
            create_session_summary=True,
        ),
        show_tool_calls=False,
        markdown=True
    )
