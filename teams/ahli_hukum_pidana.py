from typing import List, Optional
from textwrap import dedent
from pathlib import Path
import os

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.models.google import Gemini
from agno.models.deepseek import DeepSeek
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.storage.postgres import PostgresStorage
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.pgvector import PgVector
from agno.knowledge.text import TextKnowledgeBase
from agno.embedder.openai import OpenAIEmbedder
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.memory.v2.memory import Memory
from db.session import db_url
from teams.settings import team_settings
from pydantic import BaseModel, Field
from agno.tools.reasoning import ReasoningTools

COLLECTION_SOP = "wassidik"
knowledge_base_sop = TextKnowledgeBase(
    path=Path("data/wassidik"),
    vector_db = Qdrant(
        collection=COLLECTION_SOP,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        api_key=os.getenv("QDRANT_API_KEY"),
    )
)
knowledge_base_umum = TextKnowledgeBase(
    path=Path("data/krimum/umum"),  # Folder berisi dokumen terkait Dit Reskrimum
    vector_db=PgVector(
        table_name="text_dit_reskrimum",
        db_url=db_url,
        embedder=OpenAIEmbedder(),
    ),
)
COLLECTION_NAME_khusus = "tipidter"

# Inisialisasi basis pengetahuan teks yang berisi dokumen-dokumen terkait hukum untuk Tipidter
knowledge_base_khusus = TextKnowledgeBase(
    path=Path("data/tipidter"),
    vector_db = Qdrant(
        collection=COLLECTION_NAME_khusus,
        url="https://2b6f64cd-5acd-461b-8fd8-3fbb5a67a597.europe-west3-0.gcp.cloud.qdrant.io:6333",
        embedder=OpenAIEmbedder(),
        api_key=os.getenv("QDRANT_API_KEY")
    )
)

# --- Define Response Model ---
class LaporanAnalisisHukumPidana(BaseModel):
    """Model for criminal law analysis report."""
    ringkasan_eksekutif: str = Field(
        ...,
        description="Ringkasan eksekutif dari analisis kasus hukum pidana"
    )
    analisis_unsur_pidana: dict = Field(
        ...,
        description="Analisis unsur-unsur pidana yang terpenuhi dalam kasus",
        additionalProperties=False
    )
    dasar_hukum: List[str] = Field(
        ...,
        description="Daftar dasar hukum yang relevan dengan kasus"
    )
    prosedur_penyidikan: List[str] = Field(
        ...,
        description="Prosedur penyidikan yang harus dilakukan berdasarkan SOP"
    )
    rekomendasi_tindakan: List[str] = Field(
        ...,
        description="Rekomendasi tindakan hukum yang dapat diambil"
    )
    precedent_kasus: List[str] = Field(
        ...,
        description="Kasus-kasus serupa yang dapat dijadikan precedent"
    )
    risiko_hukum: Optional[str] = Field(
        None,
        description="Analisis risiko hukum dalam penanganan kasus"
    )

# Initialize shared Redis memory for the team
def create_shared_memory(model_id: str = "gpt-4o-mini") -> Memory:
    """Create shared Redis memory for the team and all agents."""
    return Memory(
        model=OpenAIChat(id=model_id),
        db=RedisMemoryDb(
            prefix="ahli_hukum_pidana_memory", 
            host=os.getenv("REDIS_HOST", "localhost"), 
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", None),
            db=int(os.getenv("REDIS_DB", "0"))
        ),
        delete_memories=True,
        clear_memories=True,
    )

# Create agents with shared memory - following agentic context pattern
def create_kb_umum_agent(memory: Memory) -> Agent:
    return Agent(
        name="Tindak Pidana Umum",
        agent_id="kb-umum-agent",
        role="Mencari informasi hukum pidana dari knowledge base umum dan peraturan perundang-undangan dasar.",
        model=OpenAIChat(
            id="gpt-4o-mini",
            max_tokens=team_settings.default_max_completion_tokens,
            temperature=team_settings.default_temperature,
        ),
        tools=[],  # Will use internal knowledge base
        instructions=[
            "Anda adalah ahli hukum pidana dengan spesialisasi dalam peraturan perundang-undangan umum!",
            "Tugas utama Anda adalah:",
            "1. Menganalisis kasus berdasarkan KUHP, KUHAP, dan peraturan dasar lainnya",
            "2. Mencari pasal-pasal yang relevan dengan unsur pidana dalam kasus",
            "3. Mengidentifikasi jenis tindak pidana yang terjadi",
            "4. Memberikan dasar hukum yang kuat untuk proses penyidikan",
            "5. Menganalisis ancaman pidana dan sanksi yang dapat dijatuhkan",
            "6. Memastikan semua unsur pidana terpenuhi sebelum melanjutkan proses hukum"
        ],
        knowledge=knowledge_base_umum,
        memory=memory,
    )

def create_kb_khusus_agent(memory: Memory) -> Agent:
    return Agent(
        name="Tindak Pidana Khusus",
        agent_id="kb-khusus-agent",
        role="Mencari informasi dari knowledge base khusus seperti tipidkor, narkotika, tipidter, dan bidang khusus lainnya.",
        model=OpenAIChat(
            id="gpt-4o-mini",
            max_tokens=team_settings.default_max_completion_tokens,
            temperature=team_settings.default_temperature,
        ),
        tools=[],  # Will use specialized knowledge base
        instructions=[
            "Anda adalah spesialis hukum pidana khusus dengan keahlian mendalam dalam berbagai bidang!",
            "Tugas utama Anda adalah:",
            "1. Menganalisis kasus berdasarkan UU khusus (Tipidkor, Narkotika, Tipidter, ITE, dll)",
            "2. Mencari regulasi spesifik yang berlaku untuk jenis kejahatan tertentu",
            "3. Mengidentifikasi modus operandi yang umum dalam kejahatan khusus",
            "4. Memberikan insight tentang penanganan kasus serupa di masa lalu",
            "5. Menganalisis aspek teknis dan forensik yang diperlukan",
            "6. Memberikan panduan investigasi yang spesifik untuk jenis kejahatan"
        ],
        knowledge=knowledge_base_khusus,
        memory=memory,
    )

def create_sop_agent(memory: Memory) -> Agent:
    return Agent(
        name="Perkaba Polri",
        agent_id="sop-agent",
        role="Mencari dan menganalisis Standard Operating Procedure (SOP) untuk proses penyidikan.",
        model=OpenAIChat(
            id="gpt-4o-mini",
            max_tokens=team_settings.default_max_completion_tokens,
            temperature=team_settings.default_temperature,
        ),
        tools=[],  # Will use SOP knowledge base
        instructions=[
            "Anda adalah ahli prosedur penyidikan dengan pengetahuan mendalam tentang SOP Polri!",
            "Tugas utama Anda adalah:",
            "1. Mengidentifikasi SOP yang relevan untuk jenis kasus yang sedang ditangani",
            "2. Memberikan langkah-langkah penyidikan yang harus dilakukan secara berurutan",
            "3. Memastikan compliance dengan prosedur internal Polri",
            "4. Mengidentifikasi dokumen dan bukti yang harus dikumpulkan",
            "5. Memberikan timeline dan deadline untuk setiap tahap penyidikan",
            "6. Memastikan semua aspek administratif dan legal terpenuhi"
        ],
        knowledge=knowledge_base_sop,
        memory=memory,
    )

def create_internet_research_agent(memory: Memory) -> Agent:
    return Agent(
        name="Penyelidik Internet",
        agent_id="internet-research-agent",
        role="Melakukan riset di internet untuk mencari precedent kasus, perkembangan hukum terkini, dan informasi pendukung.",
        model=OpenAIChat(
            id="gpt-4o-mini",
            max_tokens=team_settings.default_max_completion_tokens,
            temperature=team_settings.default_temperature,
        ),
        tools=[GoogleSearchTools(fixed_language="id"), Newspaper4kTools()],
        instructions=[
            "Anda adalah peneliti hukum digital dengan kemampuan riset internet yang excellent!",
            "Tugas utama Anda adalah:",
            "1. Mencari kasus-kasus serupa yang telah diputus pengadilan",
            "2. Mengumpulkan perkembangan hukum dan yurisprudensi terkini",
            "3. Mencari artikel akademis dan analisis hukum yang relevan",
            "4. Mengidentifikasi trend dan pola dalam penanganan kasus serupa",
            "5. Mengumpulkan informasi tentang modus operandi terbaru",
            "6. Memberikan konteks sosial dan mediatic dari kasus yang sedang ditangani"
        ],
        memory=memory,
        add_datetime_to_instructions=True,
    )

def get_ahli_hukum_pidana_team(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False
) -> Team:
    """
    Initializes and returns the Criminal Law Expert Team in route mode with Redis memory.
    
    This team consists of four specialized agents working in route mode where
    the team leader decides which agent to use for each query.
    All agents share Redis memory for enhanced context awareness.
    
    Args:
        model_id: Model ID to use for the team leader
        user_id: User ID for session tracking
        session_id: Session ID for conversation tracking
        debug_mode: Enable debug mode for more verbose logging
        
    Returns:
        Team: A route-mode Team instance for criminal law analysis with Redis memory
    """
    model_id = model_id or "gemini-2.0-flash"
    
    # Create shared memory instance
    shared_memory = create_shared_memory(model_id="gpt-4o-mini")
    
    # Create agents with shared memory
    kb_umum_agent = create_kb_umum_agent(shared_memory)
    kb_khusus_agent = create_kb_khusus_agent(shared_memory)
    sop_agent = create_sop_agent(shared_memory)
    internet_research_agent = create_internet_research_agent(shared_memory)
    
    return Team(
        name="Tim Ahli Hukum Pidana",
        team_id="ahli-hukum-pidana-team",
        mode="coordinate",  # Using coordinate mode where team leader delegates tasks to specialists
        model=DeepSeek(id="deepseek-chat"),
        tools=[
        ReasoningTools(add_instructions=True, add_few_shot=True),
    ],
        members=[kb_umum_agent, kb_khusus_agent, sop_agent, internet_research_agent],
        description="Tim ahli hukum pidana yang mengoordinasikan spesialis untuk memberikan analisis hukum pidana yang komprehensif.",
        instructions=[
            "Analisis pertanyaan hukum pidana dan delegasikan tugas ke spesialis yang tepat.",
            "Koordinasikan berdasarkan kebutuhan analisis:",
            "- Agen Tindak Pidana Umum: untuk analisis KUHP, KUHAP, dan peraturan dasar",
            "- Agen Tindak Pidana Khusus: untuk UU khusus (Tipidkor, Narkotika, Tipidter, ITE, dll)",
            "- Agen Perkaba Polri: untuk prosedur penyidikan dan compliance SOP Polri",
            "- Penyelidik Internet: untuk riset precedent dan perkembangan hukum terkini",
            "Sintesis semua input dari spesialis menjadi respons yang kohesif dan komprehensif.",
            "Pastikan analisis yang diberikan akurat dan berdasarkan sumber yang tepat.",
            "Integrasikan SEMUA informasi hukum atau kasus yang didapat dari anggota tim ke dalam analisis final."
            "Jika pengguna hanya bercakap-cakap, berikan jawaban langsung tanpa menggunakan tools."
        ],
        success_criteria="Tim telah memberikan analisis hukum pidana yang komprehensif dengan informasi dari semua spesialis yang relevan.",
        session_id=session_id,
        user_id=user_id,
        markdown=True,
        show_tool_calls=True,
        show_members_responses=False,  # Hide individual member responses in coordinate mode for cleaner output
        memory=shared_memory,  # Team also uses shared memory
        enable_agentic_context=True,  # Enable agentic context like in documentation
        share_member_interactions=True,  # Members see previous member interactions for better coordination
        storage=PostgresStorage(
            table_name="ahli_hukum_pidana_team",
            db_url=db_url,
            mode="team",
            auto_upgrade_schema=True,
        ),
        enable_team_history=True,
        debug_mode=debug_mode,
        add_datetime_to_instructions=True,
    )
