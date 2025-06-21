# main.py  ──────────────────────────────────────────────────────────────────────
import os, asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from functools import wraps

# Agno imports
from agno.agent import Agent
from agno.knowledge.light_rag import LightRagKnowledgeBase
from agno.models.google import Gemini
from agno.document import Document
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.postgres import PostgresStorage

from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.thinking import ThinkingTools

from db.session import db_url        # koneksi DB Anda

# ─── ENV & konstanta ───────────────────────────────────────────────────────────
load_dotenv()
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "https://rag.reserse.id")
LIGHTRAG_KEY = os.getenv("LIGHTRAG_API_KEY")
KUHAP_DIR    = Path("data/kuhap")

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": LIGHTRAG_KEY,
}

# ─── Memory & storage ──────────────────────────────────────────────────────────
memory = Memory(db=PostgresMemoryDb(
    table_name="kuhap_agent_memories", db_url=db_url
))
kuhap_agent_storage = PostgresStorage(
    table_name="kuhap_agent_memory", db_url=db_url
)

# ─── Custom Knowledge Base (header API-key otomatis) ───────────────────────────
class LightragKB(LightRagKnowledgeBase):
    lightrag_server_url: str = LIGHTRAG_URL

    async def _insert_text(self, text: str):
        import httpx
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self.lightrag_server_url}/documents/text",
                json={"text": text},
                headers=HEADERS,
            ); r.raise_for_status()

    async def async_search(self, query: str, **_) -> List[Document]:
        import httpx
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self.lightrag_server_url}/query",
                json={"query": query, "mode": "hybrid"},
                headers=HEADERS,
            ); r.raise_for_status()
            data = r.json()
        content = data["response"] if isinstance(data, dict) else str(data)
        return [Document(content=content, meta_data={"query": query})]

knowledge_base = LightragKB()

# ─── NEW: custom retriever yang kompatibel ─────────────────────────────────────
async def custom_retriever(
    query: str,
    num_documents: int = 5,
    mode: str = "hybrid",
) -> List[Dict[str, Any]]:
    """
    Cari ke LightRAG Server dan kembalikan list dict dokumen.
    """
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            f"{LIGHTRAG_URL}/query",
            json={"query": query, "mode": mode},
            headers=HEADERS,
        ); r.raise_for_status()
        data = r.json()

    # format menjadi list[dict] sesuai ekspektasi Agno
    if isinstance(data, dict) and "response" in data:
        data = [data["response"]]

    return [
        {"content": str(item), "source": "lightrag", "metadata": {"query": query, "mode": mode}}
        for item in (data if isinstance(data, list) else [data])
    ]

# ─── Seed dokumen sekali (opsional) ────────────────────────────────────────────
async def seed_kb(recreate: bool = False):
    if recreate:
        await knowledge_base.delete_all()

    for fp in KUHAP_DIR.rglob("*"):
        if fp.suffix.lower() not in {".txt", ".md"}:
            continue
        await knowledge_base.load_text(fp.read_text(encoding="utf-8"))
    print("Upload selesai – tunggu status FINISHED di WebUI.")

# ─── Factory Agent ────────────────────────────────────────────────────────────
def get_kuhap_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    return Agent(
        name="Ahli Hukum Acara Pidana (KUHAP)",
        agent_id="kuhap-chat",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),

        knowledge=knowledge_base,
        retriever=custom_retriever,        # ← pakai retriever baru
        search_knowledge=True,

        tools=[ThinkingTools(add_instructions=True), TavilyTools(), Newspaper4kTools()],
        storage=kuhap_agent_storage,
        memory=memory,
        description="Anda adalah ahli hukum KUHAP.",
        instructions=[
            "**PERSONA AKADEMIK:**",
            "Anda adalah seorang Doktor Hukum (Dr.) dengan spesialisasi mendalam dalam Hukum Acara Pidana (KUHAP). "
            "Sebagai akademisi senior dengan pengalaman puluhan tahun, Anda memiliki pemahaman komprehensif tentang:",
            "• Filosofi dan ratio legis setiap pasal dalam KUHAP",
            "• Perkembangan yurisprudensi dan putusan landmark pengadilan",
            "• Perbandingan sistem peradilan pidana dengan negara lain",
            "• Analisis kritis terhadap kelemahan dan kelebihan sistem KUHAP",
            "• Hubungan KUHAP dengan peraturan perundang-undangan lainnya",
            "",
            "**GAYA KOMUNIKASI AKADEMIK:**",
            "• Gunakan terminologi hukum yang tepat dan presisi",
            "• Berikan analisis yang mendalam dengan pendekatan dogmatik, normatif, dan empiris",
            "• Sertakan perspektif historis dan komparatif ketika relevan",
            "• Jelaskan implikasi praktis dari setiap ketentuan hukum",
            "• Berikan contoh kasus nyata atau hipotetis untuk memperjelas konsep",
            "",
            "**METODE ANALISIS:**",
            "• Mulai dengan analisis gramatikal (penafsiran harfiah)",
            "• Lanjutkan dengan analisis sistematis (konteks dalam keseluruhan UU)",
            "• Berikan analisis teleologis (tujuan dan maksud pembuat UU)",
            "• Pertimbangkan aspek historis dan sosiologis jika diperlukan",
            "",
            "**Prioritas:** Cari di knowledge base LightRAG lebih dulu.",
            "",
            "**Referensi Pasal:** Sertakan kutipan KUHAP dengan format:",
            "• Pasal [nomor] KUHAP: '[bunyi pasal lengkap]'",
            "• Jelaskan maksud dan tujuan pasal tersebut",
            "• Berikan interpretasi akademik yang mendalam",
            "• Hubungkan dengan pasal-pasal terkait lainnya",
            "",
            "**STRUKTUR JAWABAN AKADEMIK:**",
            "1. **Definisi dan Konsep Dasar** - Jelaskan terminologi dengan presisi akademik",
            "2. **Landasan Hukum** - Kutip pasal-pasal KUHAP yang relevan", 
            "3. **Analisis Yuridis** - Berikan interpretasi mendalam dengan berbagai metode penafsiran",
            "4. **Yurisprudensi** - Sebutkan putusan-putusan penting jika ada",
            "5. **Implikasi Praktis** - Jelaskan penerapan dalam praktik peradilan",
            "6. **Kritik dan Saran** - Berikan evaluasi kritis jika diperlukan",
            "",
            "Jawablah dengan kedalaman akademik setingkat disertasi doktoral, namun tetap dapat dipahami praktisi hukum."
        ],
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        markdown=True,
        debug_mode=debug_mode,
    )

# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # asyncio.run(seed_kb())   # ← jalankan sekali untuk unggah dokumen
    agent = get_kuhap_agent()
    print(asyncio.run(agent.aprint_response(
        "Jelaskan asas praduga tak bersalah dalam KUHAP!"
    )))
