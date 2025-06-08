import os
from typing import Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.crawl4ai import Crawl4aiTools
from agno.tools.google_maps import GoogleMapTools

def get_maps_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="Agen Pemetaan",
        agent_id="maps-agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[
        GoogleMapTools(),  # For  on Google Maps
        Crawl4aiTools(max_length=5000),  # For scraping business websites
    ],
        description="Anda adalah spesialis informasi lokasi dan bisnis yang dapat membantu dengan berbagai pertanyaan terkait pemetaan dan lokasi.",
        instructions=[
            "Ketika diberikan pertanyaan pencarian:",
            "1. Gunakan metode Google Maps yang sesuai berdasarkan jenis pertanyaan",
            "2. Untuk pencarian tempat, gabungkan data Maps dengan data website jika tersedia",
            "3. Format respons dengan jelas dan berikan detail yang relevan berdasarkan pertanyaan",
            "4. Tangani kesalahan dengan baik dan berikan umpan balik yang bermakna",
        ],
        markdown=True,
        debug_mode=debug_mode,
        structured_outputs=True,
        show_tool_calls=True,
    )
