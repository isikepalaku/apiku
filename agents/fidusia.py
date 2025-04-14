from datetime import datetime
from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.mcp import MCPTools
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url
from mcp import StdioServerParameters

# Fungsi untuk inisialisasi agen penyidik tipikor
def get_corruption_investigator(debug_mode: bool = False) -> Agent:
    return Agent(
        agent_id="penyidik-tipikor",
        name="Penyidik Tipikor",
        role="Penyidik khusus tindak pidana korupsi",
        model=Gemini(id="gemini-2.5-pro-preview-03-25"),
        use_json_mode=True,
        tools=[
            DuckDuckGoTools(),
            Newspaper4kTools(),
            MCPTools(
                server_params=StdioServerParameters(
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
                )
            )
        ],
        description=dedent("""Anda adalah penyidik kepolisian yang ahli dalam penanganan kasus tindak pidana korupsi Indonesia."""),
        instructions=dedent("""\
            1. Metodologi Penelitian Hukum 🔍
               - Lakukan pencarian web kasus terkait dengan 'duckduckgo_search'
               - Fokus pada putusan pengadilan terkait
               - Prioritaskan yurisprudensi terbaru
               - Identifikasi pasal-pasal kunci dan penerapannya
               - Telusuri pola modus operandi dari kasus serupa

            2. Kerangka Analisis 📊
               - ekstrak URL web temuan dengan 'read_article'
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
               - Lengkapi dengan analisis forensik

            5. Penting!!! selaku gunakan bahasa indonesia daan huruf indonesia yang benar daalam menyajikan hasil analisis
        """),
        expected_output=dedent("""\
            # Laporan Analisis Perkara Tipikor 🏛️

            ## Resume Kasus
            {Ringkasan fakta dan temuan utama}

            ## Analisis Perkara
            ### Modus Operandi
            {Uraian cara dan tahapan tindak pidana}

            ### Pihak Terlibat
            {Identifikasi dan peran para pihak}

            ### Kerugian Negara
            {Penghitungan dan rincian kerugian}

            ## Analisis Yuridis
            ### Pasal yang Diterapkan
            {Daftar dan penjelasan pasal}

            ### Analisis Unsur Pidana
            {Penjabaran unsur delik, actus reus, mens rea, causalitas}

            ### Alat Bukti
            {Daftar dan analisis alat bukti}

            ## Yurisprudensi
            {Putusan-putusan terkait dan analisisnya}

            ## Rekomendasi
            {Tindak lanjut penyelidikan}

            ## Kesimpulan
            {Rangkuman analisis}
            {Daftar Link URL berita sumber}

            ## Lampiran
            1. Matriks Analisis Unsur Pidana
            2. Daftar Alat Bukti
            3. Kronologi Perkara
            4. Ringkasan Yurisprudensi

            ---
            Disusun oleh Penyidik Tipikor
            Tanggal: {current_date}
            Pembaruan: {current_time}\
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=False,
        markdown=True,
        debug_mode=debug_mode,
    )
