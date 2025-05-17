import json
from typing import Iterator, List, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.tavily import TavilyTools
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
    credibility: str = Field(..., description="Tingkat kredibilitas sumber (high/medium/low)")

class SentimentData(BaseModel):
    sentiment_score: str = Field(..., description="Skor sentimen (positive/negative/neutral)")
    intensity: str = Field(..., description="Intensitas sentimen (high/medium/low)")
    key_phrases: List[str] = Field(..., description="Frasa kunci yang mencerminkan sentimen")
    context: str = Field(..., description="Konteks pembahasan")

class WebContentAnalysis(BaseModel):
    sources: List[SentimentSource] = Field(..., description="Daftar sumber konten yang dianalisis")
    total_sources: int = Field(..., description="Total jumlah sumber yang dianalisis")
    source_distribution: str = Field(..., description="Distribusi jenis sumber konten")
    search_keywords: List[str] = Field(..., description="Kata kunci yang digunakan dalam pencarian")

class SentimentAnalysis(BaseModel):
    overall_sentiment: str = Field(..., description="Sentimen keseluruhan")
    sentiment_distribution: str = Field(..., description="Distribusi sentimen")
    key_topics: List[str] = Field(..., description="Topik utama yang dibahas")
    sentiment_data: List[SentimentData] = Field(..., description="Data sentimen terperinci")

class SentimentAnalysisSystem(Workflow):
    web_analyzer: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        tools=[TavilyTools()],
        instructions=[
            "Penelusuran mendalam konten web untuk analisis sentimen:",
            "1. Cari berbagai sumber dari berita resmi, media sosial, forum, dan blog",
            "2. Pastikan keseimbangan antara sumber berita formal dan konten media sosial",
            "3. Prioritaskan sumber dengan engagement tinggi dan kredibilitas baik",
            "4. Identifikasi berbagai sudut pandang (positif, negatif, netral)",
            "5. Catat URL, judul, platform, dan kredibilitas dari setiap sumber",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=WebContentAnalysis,
        structured_outputs=True,
        debug_mode=False,
    )

    sentiment_analyzer: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        tools=[Newspaper4kTools()],
        instructions=[
            "Analisis sentimen mendalam dari konten yang ditemukan.",
            "Evaluasi tone, konteks, dan nuansa pembahasan dari setiap sumber.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=SentimentAnalysis,
        structured_outputs=True,
        debug_mode=False,
    )

    reporter: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        instructions=[
            "Buat laporan analisis sentimen komprehensif dengan struktur berikut:",
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
            "Catatan: Jika diperlukan visualisasi data, gunakan tabel untuk menyajikan data secara struktural karena aplikasi belum mendukung fitur chart.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=False,
    )

    def get_web_analysis(self, topic: str) -> Optional[WebContentAnalysis]:
        try:
            response: RunResponse = self.web_analyzer.run(
                f"Lakukan penelusuran mendalam untuk topik: {topic}"
            )
            
            if not response or not response.content:
                logger.warning("Empty Web Analysis response")
                return None
                
            if not isinstance(response.content, WebContentAnalysis):
                logger.warning("Invalid response type")
                return None
                
            return response.content
            
        except Exception as e:
            logger.warning(f"Failed: {str(e)}")
            return None

    def get_sentiment_analysis(
        self, topic: str, web_analysis: WebContentAnalysis
    ) -> Optional[SentimentAnalysis]:
        agent_input = {"topic": topic, **web_analysis.model_dump()}
        
        try:
            response: RunResponse = self.sentiment_analyzer.run(
                json.dumps(agent_input, indent=4)
            )
            
            if not response or not response.content:
                logger.warning("Empty Sentiment Analysis response")
                return None
                
            if not isinstance(response.content, SentimentAnalysis):
                logger.warning("Invalid response type")
                return None
                
            return response.content
            
        except Exception as e:
            logger.warning(f"Failed: {str(e)}")
            return None

    def get_final_report(
        self, topic: str, web_analysis: WebContentAnalysis, sentiment_analysis: SentimentAnalysis
    ) -> Optional[str]:
        agent_input = {
            "topic": topic,
            **web_analysis.model_dump(),
            **sentiment_analysis.model_dump()
        }
        
        try:
            response: RunResponse = self.reporter.run(
                json.dumps(agent_input, indent=4)
            )
            
            if not response or not response.content:
                logger.warning("Empty Final Report response")
                return None
                
            return response.content
            
        except Exception as e:
            logger.warning(f"Failed: {str(e)}")
            return None

    def run(self, topic: str) -> Iterator[RunResponse]:
        logger.info(f"Generating a sentiment analysis report for: {topic}")
        
        # Step 1: Web Content Analysis
        web_analysis = self.get_web_analysis(topic)
        if web_analysis is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not analyze web content for: {topic}"
            )
            return
            
        # Step 2: Sentiment Analysis
        sentiment_analysis = self.get_sentiment_analysis(topic, web_analysis)
        if sentiment_analysis is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Sentiment analysis failed"
            )
            return
            
        # Step 3: Generate Final Report
        final_report = self.get_final_report(topic, web_analysis, sentiment_analysis)
        if final_report is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Final report generation failed"
            )
            return
            
        # Return final report
        yield RunResponse(
            content=final_report,
            event=RunEvent.workflow_completed
        )

def get_sentiment_analyzer(debug_mode: bool = False) -> SentimentAnalysisSystem:
    """Create and configure the sentiment analysis workflow instance."""
    return SentimentAnalysisSystem(
        workflow_id="sentiment-analysis-system",
        description="Sistem Analisis Sentimen Publik",
        storage=PostgresStorage(
            table_name="sentiment_analysis_workflows",
            db_url=db_url,
            auto_upgrade_schema=True,
        ),
        debug_mode=debug_mode,
    )

# Run workflow directly if executed as script
if __name__ == "__main__":
    from rich.prompt import Prompt
    from agno.utils.pprint import pprint_run_response

    topic = Prompt.ask(
        "[bold]Masukkan topik untuk analisis sentimen[/bold]\n✨",
        default="Kebijakan transportasi publik"
    )

    url_safe_topic = topic.lower().replace(" ", "-")
    
    # Create the workflow
    analysis_system = get_sentiment_analyzer()
    
    # Set a unique session ID for this run
    analysis_system.session_id = f"sentiment-analysis-{url_safe_topic}"
    
    # Run the workflow
    hasil_analisis = analysis_system.run(topic=topic)

    # Print the results
    pprint_run_response(hasil_analisis, markdown=True)
