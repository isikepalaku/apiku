import json
from typing import Iterator, List, Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.storage.workflow.postgres import PostgresWorkflowStorage
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field
from workflows.settings import workflow_settings
from db.session import db_url

class SentimentSource(BaseModel):
    url: str = Field(..., description="URL sumber konten")
    platform: str = Field(..., description="Platform atau jenis media (news/social/blog)")
    title: str = Field(..., description="Judul atau nama konten")
    content: Optional[str] = Field(None, description="Konten yang dianalisis")
    engagement: Optional[str] = Field(None, description="Metrik engagement jika tersedia")

class SentimentData(BaseModel):
    sentiment_score: str = Field(..., description="Skor sentimen (positive/negative/neutral)")
    intensity: str = Field(..., description="Intensitas sentimen (high/medium/low)")
    key_phrases: List[str] = Field(..., description="Frasa kunci yang mencerminkan sentimen")
    context: str = Field(..., description="Konteks pembahasan")
    timestamp: str = Field(..., description="Waktu publikasi/diskusi")

class WebContentAnalysis(BaseModel):
    sources: List[SentimentSource] = Field(..., description="Daftar sumber konten yang dianalisis")
    total_sources: int = Field(..., description="Total jumlah sumber yang dianalisis")
    source_distribution: str = Field(..., description="Distribusi jenis sumber konten")
    time_range: str = Field(..., description="Rentang waktu data yang dianalisis")

class SentimentAnalysis(BaseModel):
    overall_sentiment: str = Field(..., description="Sentimen keseluruhan")
    sentiment_distribution: str = Field(..., description="Distribusi sentimen")
    key_topics: List[str] = Field(..., description="Topik utama yang dibahas")
    sentiment_data: List[SentimentData] = Field(..., description="Data sentimen terperinci")
    regional_insights: str = Field(..., description="Wawasan berdasarkan region")

class TrendAnalysis(BaseModel):
    sentiment_trends: str = Field(..., description="Tren perubahan sentimen")
    influential_factors: str = Field(..., description="Faktor-faktor yang mempengaruhi sentimen")
    temporal_patterns: str = Field(..., description="Pola berdasarkan waktu")
    future_projections: str = Field(..., description="Proyeksi tren ke depan")
    recommendations: str = Field(..., description="Rekomendasi tindakan")

class SentimentAnalysisSystem(Workflow):
    web_analyzer: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        instructions=[
            "Penelusuran mendalam konten web untuk analisis sentimen:",
            "1. Cari 5-10 sumber berita, gabungan social media dan forum terkait topik",
            "2. Prioritaskan konten dengan engagement tinggi",
            "3. Identifikasi sumber dari berbagai platform",
            "4. Evaluasi kredibilitas dan relevansi sumber",
            "5. Ekstrak data engagement dan metrics",
        ],
        tools=[GoogleSearchTools(fixed_language="id"), Newspaper4kTools()],
        add_datetime_to_instructions=True,
        response_model=WebContentAnalysis,
        structured_outputs=True,
    )

    sentiment_analyzer: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        instructions=[
            "Analisis sentimen mendalam dari konten yang ditemukan:",
            "1. Evaluasi tone dan konteks pembahasan",
            "2. Ukur intensitas sentimen per sumber",
            "3. Identifikasi frasa kunci dan topik",
            "4. Analisis variasi regional/demografis",
            "5. Hitung distribusi sentimen overall",
        ],
        add_datetime_to_instructions=True,
        response_model=SentimentAnalysis,
        structured_outputs=True,
    )

    trend_analyzer: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        instructions=[
            "Analisis tren dan pola sentimen:",
            "1. Identifikasi perubahan sentimen overtime",
            "2. Analisis faktor-faktor pengaruh",
            "3. Evaluasi pola temporal dan musiman",
            "4. Proyeksikan tren ke depan",
            "5. Rumuskan rekomendasi tindakan",
        ],
        add_datetime_to_instructions=True,
        response_model=TrendAnalysis,
        structured_outputs=True,
    )

    reporter: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        instructions=[
        "Buat Laporan Analisis Sentimen Komprehensif dengan Struktur Sebagai Berikut:",
        "1. **Ringkasan Eksekutif:**",
        "   - Sajikan gambaran umum dari temuan analisis sentimen.",
        "   - Soroti informasi kunci dan insight utama yang diperoleh.",
        "2. **Analisis Detail:**",
        "   - Berikan penjelasan mendalam mengenai distribusi sentimen (positif, negatif, dan netral) dengan dukungan data kuantitatif.",
        "   - Sertakan contoh atau kutipan penting dari data (misalnya, komentar atau review) yang mendukung hasil analisis.",
        "3. **Gambaran Tren dan Pola Sentimen:**",
        "   - Jelaskan tren dan pola sentimen yang muncul seiring waktu.",
        "4. **Rekomendasi Berdasarkan Analisis:**",
        "   - Berikan rekomendasi strategis yang dapat diambil berdasarkan hasil analisis.",
        "5. **Metodologi dan Sumber Data:**",
        "   - Jelaskan metode analisis yang digunakan dan sebutkan sumber data yang diambil.",
        "Catatan: Jika diperlukan visualisasi data, gunakan tabel untuk menyajikan data secara struktural karena aplikasi belum mendukung fitur chart."
    ],
        markdown=True,
        add_datetime_to_instructions=True,
        structured_outputs=True,
    )

    def get_web_analysis(self, topic: str) -> Optional[WebContentAnalysis]:
        try:
            response: RunResponse = self.web_analyzer.run(
                f"Analisis sentimen publik untuk topik: {topic}"
            )
            
            if not response or not response.content:
                logger.warning("Response kosong dari web analyzer")
                return None

            if isinstance(response.content, WebContentAnalysis):
                return response.content

            logger.warning("Invalid response type dari web analyzer")
            return None

        except Exception as e:
            logger.warning(f"Web analysis failed: {str(e)}")
            return None

    def get_sentiment_analysis(
        self, topic: str, web_analysis: WebContentAnalysis
    ) -> Optional[SentimentAnalysis]:
        agent_input = {
            "topic": topic,
            **web_analysis.model_dump()
        }

        try:
            response: RunResponse = self.sentiment_analyzer.run(
                json.dumps(agent_input, indent=4)
            )

            if not response or not response.content:
                logger.warning("Response kosong dari sentiment analyzer")
                return None

            if isinstance(response.content, SentimentAnalysis):
                return response.content

            logger.warning("Invalid response type dari sentiment analyzer")
            return None

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {str(e)}")
            return None

    def get_trend_analysis(
        self, topic: str,
        web_analysis: WebContentAnalysis,
        sentiment_analysis: SentimentAnalysis
    ) -> Optional[TrendAnalysis]:
        agent_input = {
            "topic": topic,
            "web_data": web_analysis.model_dump(),
            "sentiment_data": sentiment_analysis.model_dump()
        }

        try:
            response: RunResponse = self.trend_analyzer.run(
                json.dumps(agent_input, indent=4)
            )

            if not response or not response.content:
                logger.warning("Response kosong dari trend analyzer")
                return None

            if isinstance(response.content, TrendAnalysis):
                return response.content

            logger.warning("Invalid response type dari trend analyzer")
            return None

        except Exception as e:
            logger.warning(f"Trend analysis failed: {str(e)}")
            return None

    def run(self, topic: str) -> Iterator[RunResponse]:
        logger.info(f"Memulai analisis sentimen untuk: {topic}")

        # Step 1: Web Content Analysis
        web_analysis = self.get_web_analysis(topic)
        if web_analysis is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Gagal menganalisis konten web untuk: {topic}"
            )
            return

        # Step 2: Sentiment Analysis
        sentiment_analysis = self.get_sentiment_analysis(topic, web_analysis)
        if sentiment_analysis is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Gagal menganalisis sentimen"
            )
            return

        # Step 3: Trend Analysis
        trend_analysis = self.get_trend_analysis(
            topic, web_analysis, sentiment_analysis
        )

        # Step 4: Generate Final Report
        final_report: RunResponse = self.reporter.run(
            json.dumps(
                {
                    "topic": topic,
                    "web_analysis": web_analysis.model_dump(),
                    "sentiment_analysis": sentiment_analysis.model_dump(),
                    "trend_analysis": trend_analysis.model_dump() if trend_analysis else "Trend analysis not available"
                },
                indent=4
            )
        )

        # Return final report
        yield RunResponse(
            content=final_report.content,
            event=RunEvent.workflow_completed
        )

def get_sentiment_analyzer(debug_mode: bool = False) -> SentimentAnalysisSystem:
    """Create and configure the sentiment analysis workflow instance."""
    workflow = SentimentAnalysisSystem(
        workflow_id="sentiment-analysis-system",
        description="Sistem Analisis Sentimen Publik",
        session_id="sentiment-analysis",
        storage=PostgresWorkflowStorage(
            table_name="sentiment_analysis_workflows",
            db_url=db_url,
        ),
    )

    if debug_mode:
        logger.info("Mode debug aktif untuk semua agen")
        workflow.web_analyzer.debug_mode = True
        workflow.sentiment_analyzer.debug_mode = True
        workflow.trend_analyzer.debug_mode = True
        workflow.reporter.debug_mode = True

    return workflow

# Run workflow directly if executed as script
if __name__ == "__main__":
    from rich.prompt import Prompt

    topic = Prompt.ask(
        "[bold]Masukkan topik untuk analisis sentimen[/bold]\n✨",
        default="Kebijakan transportasi publik"
    )

    url_safe_topic = topic.lower().replace(" ", "-")

    analysis_system = get_sentiment_analyzer()
    analysis_system.session_id = f"sentiment-analysis-{url_safe_topic}"

    hasil_analisis = analysis_system.run(topic=topic)

    from agno.utils.pprint import pprint_run_response
    pprint_run_response(hasil_analisis, markdown=True)