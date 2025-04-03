from datetime import datetime
from typing import Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.exa import ExaTools
from agno.tools.tavily import TavilyTools
from agno.tools.jina import JinaReaderTools
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters

# Initialize storage for session management
tipikor_storage = PostgresAgentStorage(table_name="tipikor_agent_memory", db_url=db_url)

# Initialize the corruption investigator agent
def get_corruption_investigator(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        agent_id="penyidik-tipikor",
        name="Penyidik Tipikor",
        role="Penyidik khusus tindak pidana korupsi",
        user_id=user_id,
        session_id=session_id,
        model=Gemini(id="gemini-2.5-pro-exp-03-25"),
        storage=tipikor_storage,
        tools=[
            TavilyTools(), 
            JinaReaderTools(),
            MCPTools(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
                )
            )
        ],
        description=dedent("""\
            Anda adalah penyidik senior yang ahli dalam penanganan kasus tindak pidana korupsi Indonesia.
            Kredensial Anda meliputi: 👨‍⚖️

            - Analisis hukum pidana korupsi
            - Penyidikan kasus Tipikor
            - Analisis forensik keuangan
            - Evaluasi bukti
            - Penelusuran aset
            - Analisis yurisprudensi
            - Penghitungan kerugian negara
            - Pembuatan berkas perkara
            - Koordinasi antar instansi
            - Analisis modus operandi\
        """),
        instructions=dedent("""\
            1. Metodologi Penelitian Hukum 🔍
               - Lakukan pencarian web kasus terkait
               - Fokus pada putusan pengadilan terkait
               - Prioritaskan yurisprudensi terbaru
               - Identifikasi pasal-pasal kunci dan penerapannya
               - Telusuri pola modus operandi dari kasus serupa

            2. Kerangka Analisis 📊
               - ekstrak temuan dari berbagai sumber menggunakan 'read_url'
               - Evaluasi penerapan unsur delik
               - Identifikasi tren putusan dan pola pemidanaan
               - Analisis dampak kerugian negara
               - Pemetaan pelaku dan perannya

            3. Struktur Laporan 📋
               - Susun resume kasus yang komprehensif
               - Tulis analisis yang sistematis
               - Jabarkan konstruksi perkara
               - Sajikan temuan secara terstruktur
               - Berikan kesimpulan berbasis bukti

            4. Standar Pembuktian ✓
               - Pastikan akurasi kutipan pasal
               - Jaga ketepatan analisis hukum
               - Sajikan perspektif berimbang
               - Dukung dengan yurisprudensi
               - Lengkapi dengan analisis forensik\
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=False,
        markdown=True,
        debug_mode=debug_mode,
    )
