from typing import Optional

from agno.agent import Agent
from agno.storage.postgres import PostgresStorage

import google.generativeai as genai
from google.generativeai.tools import browser

from db.session import db_url

# =============================
# 1. Konfigurasi Gemini
# =============================
genai.configure(api_key="YOUR_API_KEY")  # Ganti dengan API key kamu

# =============================
# 2. Inisialisasi Gemini Model dengan Tool `browser`
# (Mirip seperti di notebook)
# =============================
gemini_model_with_browser = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=[browser],  # ini yang membuatnya bisa browsing internet
    safety_settings={
        "HARASSMENT": "BLOCK_NONE",
        "HATE": "BLOCK_NONE",
        "SEXUAL": "BLOCK_NONE",
        "DANGEROUS": "BLOCK_NONE"
    }
)

# =============================
# 3. Fungsi OSINT = Wrapper dari model.generate_content()
# (Mirip sel terakhir di notebook)
# =============================
def osint_command_tool(prompt: str) -> str:
    """
    Fungsi investigasi web (OSINT) menggunakan Gemini + browser tool.
    Mirip dengan model.generate_content() di notebook asli.
    """
    response = gemini_model_with_browser.generate_content(prompt)
    return response.text

# =============================
# 4. Agent Storage
# =============================
basic_agent_storage = PostgresStorage(
    table_name="simple_agent",
    db_url=db_url,
    auto_upgrade_schema=True
)

# =============================
# 5. Fungsi Agent
# =============================
def get_basic_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="Polisi Intelijen Web",
        role="Membantu penyelidikan dan pelacakan informasi publik secara real-time",
        agent_id="basic-agent",
        session_id=session_id,
        user_id=user_id,
        storage=basic_agent_storage,
        add_history_to_messages=True,
        num_history_responses=3,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=debug_mode,
        tools={
            "osint": osint_command_tool  # Ini yang memungkinkan command `osint ...`
        },
    )
