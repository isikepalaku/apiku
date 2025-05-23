import json
from typing import Dict, Iterator, Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field

from db.session import db_url
from workflows.settings import workflow_settings

class SentimentSource(BaseModel):
    url: str = Field(..., description="URL sumber konten")
    platform: str = Field(..., description="Platform atau jenis media (news/social/blog)")
    title: str = Field(..., description="Judul atau nama konten")
    content: Optional[str] = Field(None, description="Konten yang dianalisis")
    credibility: str = Field(..., description="Tingkat kredibilitas sumber (high/medium/low)")

class WebContentAnalysis(BaseModel):
    sources: list[SentimentSource] = Field(..., description="Daftar sumber konten yang dianalisis")

class SentimentAnalysisSystem(Workflow):
    """Advanced workflow for generating sentiment analysis reports with proper research and citations."""
    
    description: str = "Sistem Analisis Sentimen Publik yang menggunakan caching untuk performa optimal dan mencegah pengulangan task"
    
    # Search Agent: Handles intelligent web searching and source gathering
    searcher: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Penelusuran mendalam konten web untuk analisis sentimen:",
            "1. Cari 10-15 sumber dan identifikasi 5-7 yang paling relevan dan otoritatif",
            "2. Pastikan keseimbangan antara sumber berita formal dan konten media sosial", 
            "3. Prioritaskan sumber dengan engagement tinggi dan kredibilitas baik",
            "4. Identifikasi berbagai sudut pandang (positif, negatif, netral)",
            "5. Catat URL, judul, platform, dan kredibilitas dari setiap sumber",
            "Hindari sumber yang tidak otoritatif atau opinion pieces tanpa dasar.",
        ],
        response_model=WebContentAnalysis,
        structured_outputs=True,
    )

    # Content Scraper: Extracts and processes article content
    article_scraper: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        tools=[Newspaper4kTools()],
        instructions=[
            "Analisis sentimen mendalam dari konten yang ditemukan:",
            "- Ekstrak konten dari artikel menggunakan read_article",
            "- Evaluasi tone, konteks, dan nuansa pembahasan dari setiap sumber",
            "- Lewati URL atau link yang tidak valid atau tidak dapat dibaca",
            "- Pertahankan akurasi teknis dalam terminologi",
            "- Struktur konten secara logis dengan bagian yang jelas",
            "- Tangani konten paywall dengan baik",
            "Format semua dalam markdown yang bersih untuk keterbacaan optimal.",
        ],
        response_model=SentimentSource,
        structured_outputs=True,
    )

    # Content Writer Agent: Crafts engaging sentiment analysis reports from research
    writer: Agent = Agent(
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
        markdown=True,
    )

    def run(
        self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
        use_cached_report: bool = True,
    ) -> Iterator[RunResponse]:
        logger.info(f"Generating a sentiment analysis report on: {topic}")

        # Use the cached report if use_cached_report is True
        if use_cached_report:
            cached_report = self.get_cached_report(topic)
            if cached_report:
                yield RunResponse(content=cached_report, event=RunEvent.workflow_completed)
                return

        # Search the web for articles on the topic
        search_results: Optional[WebContentAnalysis] = self.get_search_results(topic, use_search_cache)

        # If no search_results are found for the topic, end the workflow
        if search_results is None or len(search_results.sources) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Sorry, could not find any articles on the topic: {topic}",
            )
            return

        # Scrape the search results
        scraped_articles: Dict[str, SentimentSource] = self.scrape_articles(
            topic, search_results, use_scrape_cache
        )

        # Prepare the input for the writer
        writer_input = {
            "topic": topic,
            "articles": [v.model_dump() for v in scraped_articles.values()],
        }

        # Run the writer and yield the response
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)

        # Save the report in the cache
        if self.writer.run_response:
            self.add_report_to_cache(topic, str(self.writer.run_response.content))

    def get_cached_report(self, topic: str) -> Optional[str]:
        logger.info("Checking if cached report exists")
        return self.session_state.get("reports", {}).get(topic)

    def add_report_to_cache(self, topic: str, report: str):
        logger.info(f"Saving report for topic: {topic}")
        self.session_state.setdefault("reports", {})
        self.session_state["reports"][topic] = report

    def get_cached_search_results(self, topic: str) -> Optional[WebContentAnalysis]:
        logger.info("Checking if cached search results exist")
        search_results = self.session_state.get("search_results", {}).get(topic)
        return (
            WebContentAnalysis.model_validate(search_results)
            if search_results and isinstance(search_results, dict)
            else search_results
        )

    def add_search_results_to_cache(self, topic: str, search_results: WebContentAnalysis):
        logger.info(f"Saving search results for topic: {topic}")
        self.session_state.setdefault("search_results", {})
        self.session_state["search_results"][topic] = search_results

    def get_cached_scraped_articles(self, topic: str):
        logger.info("Checking if cached scraped articles exist")
        scraped_articles = self.session_state.get("scraped_articles", {}).get(topic)
        return (
            SentimentSource.model_validate(scraped_articles)
            if scraped_articles and isinstance(scraped_articles, dict)
            else scraped_articles
        )

    def add_scraped_articles_to_cache(self, topic: str, scraped_articles: Dict[str, SentimentSource]):
        logger.info(f"Saving scraped articles for topic: {topic}")
        self.session_state.setdefault("scraped_articles", {})
        self.session_state["scraped_articles"][topic] = scraped_articles

    def get_search_results(
        self, topic: str, use_search_cache: bool, num_attempts: int = 3
    ) -> Optional[WebContentAnalysis]:
        # Get cached search_results from the session state if use_search_cache is True
        if use_search_cache:
            try:
                search_results_from_cache = self.get_cached_search_results(topic)
                if search_results_from_cache is not None:
                    search_results = WebContentAnalysis.model_validate(search_results_from_cache)
                    logger.info(f"Found {len(search_results.sources)} articles in cache.")
                    return search_results
            except Exception as e:
                logger.warning(f"Could not read search results from cache: {e}")

        # If there are no cached search_results, use the searcher to find the latest articles
        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.searcher.run(topic)
                if (
                    searcher_response is not None
                    and searcher_response.content is not None
                    and isinstance(searcher_response.content, WebContentAnalysis)
                ):
                    article_count = len(searcher_response.content.sources)
                    logger.info(f"Found {article_count} articles on attempt {attempt + 1}")
                    # Cache the search results
                    self.add_search_results_to_cache(topic, searcher_response.content)
                    return searcher_response.content
                else:
                    logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: Invalid response type")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")

        logger.error(f"Failed to get search results after {num_attempts} attempts")
        return None

    def scrape_articles(
        self, topic: str, search_results: WebContentAnalysis, use_scrape_cache: bool
    ) -> Dict[str, SentimentSource]:
        scraped_articles: Dict[str, SentimentSource] = {}

        # Get cached scraped_articles from the session state if use_scrape_cache is True
        if use_scrape_cache:
            try:
                scraped_articles_from_cache = self.get_cached_scraped_articles(topic)
                if scraped_articles_from_cache is not None:
                    scraped_articles = scraped_articles_from_cache
                    logger.info(f"Found {len(scraped_articles)} scraped articles in cache.")
                    return scraped_articles
            except Exception as e:
                logger.warning(f"Could not read scraped articles from cache: {e}")

        # Scrape the articles that are not in the cache
        for article in search_results.sources:
            if article.url in scraped_articles:
                logger.info(f"Found scraped article in cache: {article.url}")
                continue

            article_scraper_response: RunResponse = self.article_scraper.run(article.url)
            if (
                article_scraper_response is not None
                and article_scraper_response.content is not None
                and isinstance(article_scraper_response.content, SentimentSource)
            ):
                scraped_articles[article_scraper_response.content.url] = article_scraper_response.content
                logger.info(f"Scraped article: {article_scraper_response.content.url}")

        # Save the scraped articles in the session state
        self.add_scraped_articles_to_cache(topic, scraped_articles)
        return scraped_articles


def get_sentiment_analyzer(debug_mode: bool = False) -> SentimentAnalysisSystem:
    return SentimentAnalysisSystem(
        workflow_id="sentiment-analysis-system",
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

    # Convert the topic to a URL-safe string for use in session_id
    url_safe_topic = topic.lower().replace(" ", "-")
    
    # Create the workflow with caching enabled
    analysis_system = get_sentiment_analyzer()
    
    # Set a unique session ID for this run
    analysis_system.session_id = f"sentiment-analysis-{url_safe_topic}"
    
    # Run the workflow with caching enabled to prevent repetitive tasks
    hasil_analisis = analysis_system.run(
        topic=topic,
        use_search_cache=True,
        use_scrape_cache=True,
        use_cached_report=True
    )

    # Print the results
    pprint_run_response(hasil_analisis, markdown=True)
