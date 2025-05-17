from typing import List, Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.models.google import Gemini
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.storage.postgres import PostgresStorage
from db.session import db_url
from pydantic import BaseModel, Field

# --- Define Response Model ---
class LaporanSentimen(BaseModel):
    """Model for sentiment analysis report."""
    ringkasan_sentimen: str = Field(
        ...,
        description="Ringkasan keseluruhan temuan analisis sentimen"
    )
    distribusi_sentimen: dict = Field(
        ...,
        description="Distribusi sentimen di berbagai kategori",
        additionalProperties=False
    )
    temuan_utama: List[str] = Field(
        ...,
        description="Daftar temuan-temuan utama dari analisis"
    )
    analisis_tren: str = Field(
        ...,
        description="Analisis tren sentimen dari waktu ke waktu"
    )
    sumber_data: List[str] = Field(
        ...,
        description="Daftar sumber data yang digunakan dalam analisis"
    )
    konteks: Optional[str] = Field(
        None,
        description="Informasi kontekstual tambahan"
    )

# --- Team and Agent Settings ---
PENCARIAN_WEB_SETTINGS = {
    "name": "Web Agent",
    "id": "web-agent",
    "role": "Mencari informasi di web berdasarkan input pengguna dan membuat beberapa kueri kata kunci.",
    "instructions": [
        "Anda adalah peneliti web dan analis berita yang berpengalaman!",
        "Tugas utama Anda adalah:"
        "1. Membuat kueri kata kunci yang relevan dari masukan pengguna",
        "2. Melakukan pencarian web untuk mengumpulkan URL yang relevan",
        "3. Fokus pada sumber-sumber berita, artikel, dan diskusi online yang dapat diandalkan",
        "4. Mengumpulkan minimal 5-10 URL yang dapat memberikan informasi komprehensif tentang topik",
        "5. Menyajikan daftar URL dengan deskripsi singkat tentang apa yang ditemukan di sumber tersebut"
    ],
}

EKSTRAKSI_KONTEN_SETTINGS = {
    "name": "Content Extraction Agent",
    "id": "content-extraction-agent",
    "role": "Mengekstrak konten tekstual utama dari URL web yang disediakan.",
    "instructions": [
        "Anda adalah spesialis ekstraksi konten dengan pengalaman mendalam!",
        "Tugas utama Anda adalah:"
        "1. Membuka setiap URL yang diberikan oleh Web Agent",
        "2. Mengekstrak konten utama yang relevan dari setiap URL",
        "3. Membersihkan teks dari elemen yang tidak diperlukan (iklan, menu, dll.)",
        "4. Memastikan konten yang diekstrak adalah teks lengkap dan utuh",
        "5. Menggabungkan semua konten yang diekstrak dengan penanda sumber yang jelas"
    ],
}

PEMBUAT_LAPORAN_SETTINGS = {
    "name": "Sentiment Report Agent",
    "id": "sentiment-report-agent",
    "role": "Menganalisis sentimen dari teks yang diekstrak dan menghasilkan laporan komprehensif.",
    "instructions": [
        "Anda adalah analis sentimen profesional dengan keahlian dalam NLP!",
        "Tugas utama Anda adalah:"
        "1. Menganalisis teks yang diekstrak untuk menentukan sentimen umum",
        "2. Mengkategorikan sentimen ke dalam sentimen positif, negatif, dan netral dengan persentase",
        "3. Mengidentifikasi tema-tema utama dan pendorong opini dalam konten",
        "4. Menganalisis tren sentimen dan perubahan dari waktu ke waktu (jika data tersedia)",
        "5. Menyusun laporan analisis sentimen komprehensif sesuai dengan format yang ditentukan",
        "6. Menyertakan contoh atau kutipan penting dari data yang mendukung hasil analisis",
        "7. Berikan rekomendasi strategis berdasarkan hasil analisis"
    ],
}

TIM_SETTINGS = {
    "team_id": "sentiment-analysis-team",
    "name": "Tim Analisis Sentimen",
    "description": dedent("""\
        Tim ahli yang berkoordinasi untuk melakukan analisis sentimen mendalam terhadap topik yang ditentukan.
        Tim ini terdiri dari spesialis pencarian web, ekstraksi konten, dan analisis sentimen.
        Kredensial Tim meliputi: 📊
        - Pengumpulan data sentimen
        - Analisis sentimen kuantitatif dan kualitatif
        - Analisis tren temporal
        - Pemetaan konteks
        - Identifikasi opinion drivers
        - Analisis engagement metrics
        - Natural Language Processing
        - Visualisasi sentimen
        - Analisis platform-specific
        - Pelaporan komprehensif\
    """),
    "instructions": [
        "Anda adalah koordinator utama dari Tim Analisis Sentimen.",
        "Tujuan utama adalah menghasilkan laporan sentimen yang komprehensif berdasarkan topik yang diberikan pengguna.",
        "",
        "### Alur Kerja Tim:",
        "",
        "1. PENCARIAN WEB:",
        "   - Gunakan Web Agent untuk menemukan URL sumber yang relevan dengan topik yang diberikan",
        "   - URL harus dari berbagai jenis sumber (berita, blog, forum, media sosial)",
        "",
        "2. EKSTRAKSI KONTEN:",
        "   - Gunakan Content Extraction Agent untuk mengekstrak konten utama dari URL",
        "   - Pastikan konten bersih dan relevan dengan topik",
        "",
        "3. ANALISIS SENTIMEN:",
        "   - Gunakan Sentiment Report Agent untuk menganalisis sentimen dari konten yang diekstrak",
        "   - Hasilkan laporan dengan format berikut:",
        "",
        "### Format Laporan:",
        "",
        "# Analisis Sentimen: [Topik]",
        "",
        "## 1. Ringkasan Eksekutif",
        "   - Sajikan gambaran umum dari temuan analisis sentimen",
        "   - Soroti informasi kunci dan insight utama yang diperoleh",
        "",
        "## 2. Analisis Detail",
        "   - Berikan penjelasan mendalam mengenai distribusi sentimen (positif, negatif, dan netral) dengan dukungan data kuantitatif",
        "   - Sertakan contoh atau kutipan penting dari data (misalnya, komentar atau review) yang mendukung hasil analisis",
        "",
        "## 3. Gambaran Tren dan Pola Sentimen",
        "   - Jelaskan tren dan pola sentimen yang muncul seiring waktu",
        "",
        "## 4. Rekomendasi Berdasarkan Analisis",
        "   - Berikan rekomendasi strategis yang dapat diambil berdasarkan hasil analisis",
        "",
        "## 5. Metodologi dan Sumber Data",
        "   - Jelaskan metode analisis yang digunakan",
        "   - Sebutkan sumber data yang diambil",
        "",
        "Catatan: Jika diperlukan visualisasi data, gunakan tabel untuk menyajikan data secara struktural karena aplikasi belum mendukung fitur chart."
    ]
}

# Create the Web Search Agent
def create_web_agent():
    """Create an agent for web searches."""
    return Agent(
        name=PENCARIAN_WEB_SETTINGS["name"],
        agent_id=PENCARIAN_WEB_SETTINGS["id"],
        model=OpenAIChat("gpt-4o-mini"),
        role=PENCARIAN_WEB_SETTINGS["role"],
        tools=[GoogleSearchTools(fixed_language="id")],
        instructions=PENCARIAN_WEB_SETTINGS["instructions"],
        show_tool_calls=True,
        markdown=True,
        storage=PostgresStorage(
            table_name="web_agent", 
            db_url=db_url, 
            auto_upgrade_schema=True
        )
    )

# Create the Content Extraction Agent
def create_content_extraction_agent():
    """Create an agent for content extraction from web URLs."""
    return Agent(
        name=EKSTRAKSI_KONTEN_SETTINGS["name"],
        agent_id=EKSTRAKSI_KONTEN_SETTINGS["id"],
        model=OpenAIChat("gpt-4o-mini"),
        role=EKSTRAKSI_KONTEN_SETTINGS["role"],
        tools=[Newspaper4kTools()],
        instructions=EKSTRAKSI_KONTEN_SETTINGS["instructions"],
        show_tool_calls=True,
        markdown=True,
        storage=PostgresStorage(
            table_name="content_extraction_agent", 
            db_url=db_url, 
            auto_upgrade_schema=True
        )
    )

# Create the Sentiment Analysis & Report Agent
def create_sentiment_report_agent():
    """Create an agent for sentiment analysis and report creation."""
    return Agent(
        name=PEMBUAT_LAPORAN_SETTINGS["name"],
        agent_id=PEMBUAT_LAPORAN_SETTINGS["id"],
        model=Gemini(id="gemini-2.0-flash"),
        role=PEMBUAT_LAPORAN_SETTINGS["role"],
        tools=[],
        instructions=PEMBUAT_LAPORAN_SETTINGS["instructions"],
        show_tool_calls=True,
        markdown=True,
        storage=PostgresStorage(
            table_name="sentiment_report_agent", 
            db_url=db_url, 
            auto_upgrade_schema=True
        )
    )

def get_sentiment_analysis_team(debug_mode: bool = False) -> Team:
    """
    Initializes and returns the Sentiment Analysis Team.
    
    This team consists of three specialized agents working together in a coordinated manner:
    1. Web Agent: Searches the web for relevant sources on a given topic
    2. Content Extraction Agent: Extracts content from web URLs
    3. Sentiment Report Agent: Analyzes sentiment and creates comprehensive reports
    
    Args:
        debug_mode: Enable debug mode for more verbose logging
        
    Returns:
        Team: A coordinated Team instance for sentiment analysis
    """
    # Create all agents
    web_agent = create_web_agent()
    content_extraction_agent = create_content_extraction_agent()
    sentiment_report_agent = create_sentiment_report_agent()
    
    # Create and return the team
    return Team(
        team_id=TIM_SETTINGS["team_id"],
        name=TIM_SETTINGS["name"],
        mode="coordinate",
        model=Gemini(id="gemini-2.0-flash"),
        members=[web_agent, content_extraction_agent, sentiment_report_agent],
        storage=PostgresStorage(
            table_name="sentiment_analysis_team",
            db_url=db_url,
            mode="team",
            auto_upgrade_schema=True,
        ),
        description=TIM_SETTINGS["description"],
        instructions=TIM_SETTINGS["instructions"],
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
        show_members_responses=True,
        enable_agentic_context=True,
        enable_team_history=True
    )
