import os
from pathlib import Path
from typing import Optional

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.tavily import TavilyTools
from agno.tools.file import FileTools
from agno.tools.newspaper4k import Newspaper4kTools

def get_sentiment_team(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    # Setup temporary file for storing URLs
    urls_file = Path(__file__).parent.joinpath("tmp", f"urls__{session_id}.md")
    urls_file.parent.mkdir(parents=True, exist_ok=True)

    # Web Search Agent
    web_searcher = Agent(
        name="Penelusur Web",
        role="Mencari dan menganalisis konten web untuk sentimen publik",
        model=OpenRouter(id="google/gemini-2.0-flash-exp:free"),
        instructions=[
            "Untuk setiap topik yang diberikan:",
            "1. Buat 3 kata kunci pencarian yang relevan untuk menganalisis sentimen publik",
            "2. Lakukan pencarian mendalam untuk setiap kata kunci",
            "3. Identifikasi 10 URL terpercaya dengan fokus pada:",
            "   - Platform media sosial dengan diskusi aktif",
            "   - Forum online dengan engagement tinggi",
            "   - Artikel berita dari sumber kredibel",
            "   - Blog dan ulasan dengan dampak signifikan",
            "4. Prioritaskan sumber berdasarkan:",
            "   - Kredibilitas dan reputasi",
            "   - Tingkat engagement publik",
            "   - Kebaruan informasi",
            "   - Relevansi dengan topik"
        ],
        tools=[TavilyTools()],
        save_response_to_file=str(urls_file),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )

    # Content Analysis Agent
    content_analyzer = Agent(
        name="Penganalisis Konten",
        role="Mengekstrak dan menganalisis konten dari artikel dan diskusi",
        model=OpenRouter(id="openai/gpt-4o-mini"),
        description="Anda adalah analis senior yang ahli dalam mengekstrak dan menganalisis sentimen dari berbagai sumber konten.",
        instructions=[
            f"1. Baca semua URL yang tersimpan di {urls_file.name} menggunakan `get_article_text`",
            "2. Lakukan analisis mendalam untuk setiap konten:",
            "   - Ekstrak sentimen utama dan tone pembahasan",
            "   - Identifikasi pola opini dan argumentasi",
            "   - Ukur tingkat engagement dan reach",
            "   - Analisis tren temporal dan geografis",
            "3. Sintesis temuan menjadi laporan komprehensif:",
            "   - Minimum 15 paragraf analisis",
            "   - Sertakan data kuantitatif dan kualitatif",
            "   - Gunakan kutipan langsung untuk mendukung temuan",
            "   - Buat visualisasi distribusi sentimen",
            "4. Pastikan kualitas analisis:",
            "   - Verifikasi fakta dan sumber",
            "   - Hindari bias dan subjektivitas",
            "   - Berikan konteks yang relevan",
            "   - Tunjukkan metodologi yang digunakan"
        ],
        tools=[Newspaper4kTools(), FileTools(base_dir=urls_file.parent)],
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )

    # Lead Sentiment Analyst (Team Coordinator)
    return Agent(
        name="Tim Analisis Sentimen",
        role="Menganalisis dan melaporkan sentimen publik",
        agent_id="sentiment-team",
        model=OpenRouter(id="openai/gpt-4o-mini"),
        team=[web_searcher, content_analyzer],
        description="Anda adalah kepala tim analisis sentimen yang bertanggung jawab menghasilkan laporan sentimen komprehensif.",
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
            "{current_datetime}"
        ],
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )