import os
from datetime import datetime
from typing import List, Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.tavily import TavilyTools
from agno.tools.jina import JinaReaderTools
from agno.storage.postgres import PostgresStorage # Import storage
from db.session import db_url # Import db_url
from pydantic import BaseModel

# --- Define Response Model ---
class LaporanPenyidikan(BaseModel):
    """Struktur laporan hasil penyidikan tindak pidana korupsi."""
    ringkasan_kasus: str
    temuan_utama: List[str]
    analisis_hukum: str
    referensi: List[str]
    modus_operandi: Optional[str] = None

# --- Define Specialized Agents ---

# Agent for Web Searching
pencari_web = Agent(
    name="Pencari Web",
    # Using Gemini consistent with the original agent
    model=Groq(
        id="qwen-2.5-32b",
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    role="Mencari informasi relevan di web terkait kasus korupsi, hukum, preseden, dan berita terkini.",
    tools=[TavilyTools()],
    description="Ahli dalam menemukan informasi spesifik di web menggunakan mesin pencari.",
    add_datetime_to_instructions=True, # Keep context fresh
)

# Agent for Reading/Analyzing URLs/Documents
pembaca_dokumen = Agent(
    name="Pembaca Dokumen",
    tools=[JinaReaderTools()], # Using Jina for reading URLs
    role="Membaca dan menganalisis konten dari URL atau dokumen yang diberikan untuk mengekstrak informasi kunci.",
    description="Mampu memahami dan merangkum konten dari halaman web atau dokumen.",
)

# --- Define the Coordination Team ---

# Note: The original storage logic (PostgresAgentStorage) is removed for simplicity in this team structure.
# If session management across team interactions is needed, storage would need to be integrated,
# potentially at the Team level or passed to members if they need individual memory.

def get_corruption_investigator_team(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None, # Kept for potential future storage integration
    debug_mode: bool = False,
) -> Team:
    """
    Initializes and returns the coordinated team for corruption investigation.
    The team's team_id is set to 'penyidik-tipikor'.
    """
    return Team(
        team_id="penyidik-tipikor", # Corrected: Use team_id for Team class
        name="Tim Penyidik Tipikor",
        mode="coordinate",
        # Coordinator Model
        model=OpenAIChat(id="gpt-4o"), # Using Flash for coordination potentially faster/cheaper
        members=[pencari_web, pembaca_dokumen],
        user_id=user_id, # Pass user/session if needed later
        session_id=session_id,
        storage=PostgresStorage( # Add storage configuration
            table_name="penyidik_tipikor_team", # Using team_id as table name convention
            db_url=db_url,
            mode="team",
            auto_upgrade_schema=True,
        ),
        description=dedent("""\
            Tim ahli yang berkoordinasi untuk melakukan investigasi mendalam terhadap kasus tindak pidana korupsi di Indonesia.
            Tim ini terdiri dari spesialis pencarian web dan analisis dokumen.
            Kredensial Tim meliputi: 👨‍⚖️
            - Analisis hukum pidana korupsi
            - Penyidikan kasus Tipikor
            - Analisis forensik keuangan (melalui koordinasi)
            - Evaluasi bukti
            - Penelusuran aset (melalui koordinasi)
            - Analisis yurisprudensi
            - Penghitungan kerugian negara
            - Pembuatan laporan investigasi
            - Koordinasi antar spesialis
            - Analisis modus operandi\
        """),
        instructions=[
            "Anda adalah Koordinator Tim Penyidik Tipikor.",
            "Tugas utama Anda adalah menganalisis permintaan pengguna terkait kasus korupsi.",
            "Gunakan anggota tim untuk mengumpulkan informasi yang relevan:",
            " - Tugaskan 'Pencari Web' untuk mencari berita, data hukum, atau informasi terkait lainnya di internet.",
            " - Tugaskan 'Pembaca Dokumen' untuk membaca dan menganalisis konten dari URL atau dokumen yang ditemukan.",
            "Kumpulkan dan sintesis semua informasi yang diperoleh dari anggota tim.",
            "Lakukan analisis mendalam terhadap informasi yang terkumpul, seperti seorang penyidik korupsi profesional.",
            "Identifikasi fakta kunci, potensi pelanggaran hukum, modus operandi, dan pihak terkait.",
            "Susun laporan akhir yang komprehensif dalam format 'LaporanPenyidikan', berdasarkan analisis Anda terhadap temuan tim.",
            "Pastikan laporan terstruktur, logis, dan menjawab pertanyaan pengguna secara menyeluruh.",
            "Jika informasi kurang, instruksikan anggota tim untuk mencari atau menganalisis lebih lanjut.",
        ],
        response_model=LaporanPenyidikan, # Re-adding based on coordinate.py example and runtime error
        add_datetime_to_instructions=True, # Coordinator gets current time context
        show_tool_calls=True, # Show tool calls for debugging/transparency
        markdown=True,
        debug_mode=debug_mode,
        show_members_responses=True, # Show responses from members for clarity
        enable_agentic_context=True, # Allow context sharing between members if needed
    )
