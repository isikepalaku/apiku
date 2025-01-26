import os
import json
from typing import Optional
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.exa import ExaTools
from phi.tools.newspaper4k import Newspaper4k

def get_search_results() -> str:
    """Get current search results from state"""
    return json.dumps([])

def update_search_results(results: str, query: str) -> str:
    """
    Update search results in state
    
    Args:
        results: JSON string of search results
        query: Search query used
    """
    return f"Search results updated for query: {query}"

def get_search_status() -> str:
    """Get current search state"""
    status = {
        "results_count": 0,
        "last_query": "",
        "status": "Ready"
    }
    return json.dumps(status)

def set_search_status(status: str) -> str:
    """
    Set current search status
    
    Args:
        status: Current status message
    """
    return f"Status updated to: {status}"

def get_web_search_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="Analisis Riset Hukum SPKT",
        agent_id="web_search_agent",
        session_id=session_id,
        user_id=user_id,
        model=OpenAIChat(
            id=model_id or "gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"]
        ),
        tools=[
        ExaTools(
            api_key=os.environ["EXA_API_KEY"],
            include_domains=[
                "scholar.google.com",
                "hukumonline.com",
                "mahkamahagung.go.id",
                "kemenkumham.go.id",
                "peraturan.go.id"
            ],
            text_length_limit=1000,
            highlights=False,
            num_results=5
        ),
        Newspaper4k(),
        get_search_results,
        update_search_results,
        get_search_status,
        set_search_status
    ],
        description="Saya adalah asisten riset yang membantu penyidik menganalisis sumber-sumber hukum online untuk mendukung proses penyidikan.",
        instructions=[
            "1. Cari referensi hukum dengan ExaTools",
            "2. Ekstrak konten dengan Newspaper4k",
            "3. Analisis: yurisprudensi, peraturan, dan pendapat ahli",
            "4. Berikan rekomendasi berdasarkan temuan"
        ],
        session_state={
            "search_results": [],
            "last_query": "",
            "status": "Ready"
        },
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        stream=True,
        monitoring=True,
        debug_mode=debug_mode,
    )
