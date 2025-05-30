import os
import asyncio
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.models.google import Gemini
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.qdrant import Qdrant
from agno.storage.postgres import PostgresStorage
from db.session import db_url
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.thinking import ThinkingTools

load_dotenv()  # Load environment variables from .env file

# Initialize memory v2 and storage
memory = Memory(db=PostgresMemoryDb(table_name="tipidkor_agent_memories", db_url=db_url))
tipidkor_agent_storage = PostgresStorage(table_name="tipidkor_agent_memory", db_url=db_url)
COLLECTION_NAME = "tipidkorkb"
# Initialize text knowledge base with multiple documents
knowledge_base = TextKnowledgeBase(
    path=Path("data/tipidkor"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# Load knowledge base before initializing agent
#knowledge_base.load(recreate=False)

def get_tipidkor_agent(
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
        name="TIPIDKOR Chat",
        agent_id="tipidkor-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-05-20"),
        use_json_mode=True,
        tools=[
            ThinkingTools(add_instructions=True),
            GoogleSearchTools(cache_results=True), 
            Newspaper4kTools(),
        ],
        knowledge=knowledge_base,
        storage=tipidkor_agent_storage,
        search_knowledge=True,
        description="Penyidik kepolisian tindak pidana korupsi.",
        instructions=[
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "**Audience:** Pengguna yang bertanya kepadamu adalah penyidik yang sudah memiliki keahlian mendalam di bidang penyidikkan, jawabanmu harus teliti, akurat dan mendalam.\n",
            "Ingat selalu awali dengan pencarian di knowledge base menggunakan search_knowledge_base tool.\n",
            "Analisa semua hasil dokumen yang dihasilkan sebelum memberikan jawaban.\n",
            "Jika beberapa dokumen dikembalikan, sintesiskan informasi secara koheren.\n",
            "Lakukan pencarian internet dengan google_search jika tidak ditemukan jawaban di basis pengetahuanmu.\n",
            "Berikan panduan selayaknya mentor penyidik kepolsian yang ahli pada tindak pidana korupsi\n",
            
            "# Dasar Hukum\n"
            "- Selalu mengacu pada UU dan peraturan yang relevan\n"
            "- Menyebutkan pasal-pasal spesifik yang terkait\n"
            "- Menjelaskan interpretasi hukum secara jelas\n",
            
            "# Kategori Tindak Pidana\n"
            "- Menjelaskan jenis-jenis tindak pidana korupsi\n"
            "- Menguraikan unsur-unsur pidana yang harus terpenuhi\n"
            "- Memberikan contoh kasus yang relevan\n",
            
            "# Penjelasan Sanksi\n"
            "- Menerangkan sanksi pidana yang dapat diterapkan\n"
            "- Menjelaskan sanksi tambahan jika ada\n"
            "- Membahas precedent hukum terkait\n",
            
            "# Referensi\n"
            "- Menyertakan sumber hukum yang dirujuk\n"
            "- Mengutip yurisprudensi relevan\n",

            "# Klarifikasi dan Follow up\n"
            "- setelah membuat jawaban, follow up pertanyaan user dan berikan rekomendasi topik terkait\n"
            "- Tinjau respons Anda untuk memastikan kejelasan, kedalaman, dan keterlibatan. \n",
            "- Gunakan tabel jika memungkinkan\n",
            "- Penting, selalu gunakan bahasa indonesia dan huruf indonesia yang benar\n",
            "- ingat kamu adalah ai model bahasa besar yang dibuat khusus untuk penyidikan kepolisian\n",
        ],
        additional_context=additional_context,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        show_tool_calls=False,
        markdown=True,
        add_history_to_messages=True,
        num_history_responses=5,
        read_chat_history=True,
        memory=memory,
    )
