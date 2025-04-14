from typing import List, Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.team import Team
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
        description="Overall summary of sentiment analysis findings"
    )
    distribusi_sentimen: dict = Field(
        ...,
        description="Distribution of sentiments across categories",
        additionalProperties=False
    )
    temuan_utama: List[str] = Field(
        ...,
        description="List of key findings from the analysis"
    )
    analisis_tren: str = Field(
        ...,
        description="Analysis of sentiment trends over time"
    )
    sumber_data: List[str] = Field(
        ...,
        description="List of data sources used in analysis"
    )
    konteks: Optional[str] = Field(
        None,
        description="Additional contextual information"
    )

# --- Define Specialized Agents ---

pencari_web = Agent(
    name="Pencari Web",
    agent_id="pencari-web",
    model=OpenAIChat("gpt-4o-mini"),
    role="Mencari dan mengumpulkan data untuk analisis sentimen.",
    tools=[GoogleSearchTools(cache_results=True)],
    instructions=[
        "Cari dan kumpulkan data dari berbagai platform dan sumber online.",
        "Fokus pada konten yang relevan dengan topik analisis sentimen.",
    ],
    show_tool_calls=True,
    markdown=True,
)

pembaca_dokumen = Agent(
    name="Penganalisis Konten",
    agent_id="penganalisis-konten",
    model=OpenAIChat("gpt-4o-mini"),
    role="Menganalisis sentimen dan insights dari konten.",
    tools=[Newspaper4kTools()],
    instructions=[
        "Analisis konten untuk ekstraksi sentimen dan insights.",
        "Berikan analisis mendalam dari sumber yang diberikan.",
    ],
    show_tool_calls=True,
    markdown=True,
)

def get_sentiment_analysis_team(debug_mode: bool = False) -> Team:
    """
    Initializes and returns the coordinated team for sentiment analysis.
    """
    return Team(
        team_id="sentiment-analysis-team",
        name="Tim Analisis Sentimen",
        mode="coordinate",
        model=OpenAIChat("gpt-4o-mini"),
        members=[pencari_web, pembaca_dokumen],
        storage=PostgresStorage(
            table_name="sentiment_analysis_team",
            db_url=db_url,
            mode="team",
            auto_upgrade_schema=True,
        ),
        description=dedent("""\
            Tim ahli yang berkoordinasi untuk melakukan analisis sentimen mendalam terhadap topik yang ditentukan.
            Tim ini terdiri dari spesialis pencarian web dan analisis konten.
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
        instructions=[
            "1. Koordinasi Tim:",
            "   - Instruksikan penelusur web untuk mengumpulkan sumber-sumber relevan",
            "   - Arahkan penganalisis konten untuk melakukan analisis mendalam",
            "   - Pantau kualitas dan konsistensi temuan",
            "2. Review dan Finalisasi:",
            "   - Edit dan sempurnakan hasil analisis",
            "   - Pastikan kedalaman dan kelengkapan laporan",
            "   - Verifikasi akurasi data dan kutipan",
            "   - Tingkatkan kualitas presentasi",
            "",
            "Format Laporan Final:",
            "",
            "# Analisis Sentimen: [Topik]",
            "",
            "## Ringkasan Eksekutif",
            "- Temuan utama dan signifikansi",
            "- Tren sentimen dominan",
            "- Implikasi kunci",
            "(dalam bentuk tabel dengan persentase)",
            "",
            "## Metodologi",
            "- Sumber data dan periode analisis",
            "- Teknik pengumpulan data",
            "- Framework analisis",
            "",
            "## Analisis Sentimen Mendalam",
            "- Distribusi sentimen (kuantitatif)",
            "- Analisis temporal",
            "- Pemetaan geografis/demografis",
            "- Konteks dan faktor pengaruh",
            "",
            "## Sumber dan Platform",
            "- Breakdown engagement per platform",
            "- Kredibilitas sumber",
            "- Jangkauan dan dampak",
            "",
            "## Pola dan Tren",
            "- Narrative analysis",
            "- Opinion drivers",
            "- Emerging issues",
            "",
            "## Rekomendasi",
            "- Strategic insights",
            "- Action points",
            "- Risk mitigation",
            "",
            "Tim Analisis Sentimen",
        ],
        response_model=LaporanSentimen,
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
        show_members_responses=True,
        enable_agentic_context=True,
        enable_team_history=True
    )
