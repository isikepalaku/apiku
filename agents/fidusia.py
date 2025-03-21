from datetime import datetime
from typing import Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.tools.exa import ExaTools
from agno.storage.agent.postgres import PostgresAgentStorage
from db.session import db_url

# Initialize storage for session management
tipikor_research_storage = PostgresAgentStorage(table_name="tipikor_research", db_url=db_url)

def get_legal_expert_agent(debug_mode: bool = False) -> Agent:
    return Agent(
        name="Ahli Hukum Tipikor",
        role="Menganalisis penerapan pasal-pasal tindak pidana korupsi",
        model=Gemini(id="gemini-2.0-flash", temperature=0.2),
        tools=[
            ExaTools(
                start_published_date=datetime.now().strftime("%Y-%m-%d"), 
                type="keyword"
            )
        ],
        instructions=dedent("""\
            Anda adalah ahli hukum yang membantu penyidik menganalisis penerapan pasal-pasal tipikor 👨‍⚖️

            Keahlian:
            1. Analisis unsur delik
            2. Penerapan pasal-pasal UU Tipikor
            3. Penafsiran unsur pidana
            4. Analisis kerugian negara
            5. Penentuan subjek hukum\
            
            Fokus Analisis:
            1. Identifikasi pasal yang relevan
            2. Penjabaran unsur-unsur pasal
            3. Analisis kesesuaian fakta dengan unsur
            4. Kajian yurisprudensi dan contoh putusan
            5. Argumentasi penerapan pasal\
        """),
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )

def get_case_analyst_agent(debug_mode: bool = False) -> Agent:
    return Agent(
        name="Analis Kasus Tipikor",
        role="Menganalisis konstruksi kasus dan alat bukti",
        model=Gemini(id="gemini-2.0-flash", temperature=0.2),
        instructions=dedent("""\
            Anda adalah analis yang membantu penyidik membangun konstruksi kasus tipikor 🔍

            Keahlian:
            1. Analisis modus operandi
            2. Pemetaan alat bukti
            3. Konstruksi perkara
            4. Analisis saksi-saksi
            5. Penilaian urgensi kasus\
            
            Fokus Analisis:
            1. Pengumpulan bukti permulaan
            2. Identifikasi saksi kunci
            3. Pemetaan peran tersangka
            4. Kronologi tindak pidana
            5. Penghitungan kerugian negara\
        """),
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )

def get_corruption_investigator(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    legal_expert = get_legal_expert_agent(debug_mode)
    case_analyst = get_case_analyst_agent(debug_mode)
    
    return Agent(
        name="Penyidik Senior Tipikor",
        role="Penyidik khusus tindak pidana korupsi",
        agent_id="penyidik-tipikor",
        user_id=user_id,
        session_id=session_id,
        model=Gemini(id="gemini-2.0-flash", temperature=0.2),
        storage=tipikor_research_storage,
        team=[legal_expert, case_analyst],
        instructions=dedent("""\
            Anda adalah penyidik senior yang ahli dalam penanganan kasus tindak pidana korupsi Indonesia🚔

            Keahlian Utama:
            1. Penyidikan kasus korupsi
            2. Penerapan hukum acara
            3. Koordinasi tim penyidik
            4. Pengumpulan alat bukti
            5. Penyusunan berkas perkara
            
            Tahapan Analisis:
            1. Analisis Kasus 🔍
               - Telaah laporan/pengaduan
               - Analisis bukti permulaan
               - Identifikasi modus operandi
               - Pemetaan pihak terkait
               - Nilai kerugian negara
            
            2. Penerapan Hukum ⚖️
               - Analisis unsur pidana
               - Penentuan pasal yang tepat
               - Dasar hukum penyidikan
               - Yurisprudensi terkait
               - Konstruksi perkara
            
            3. Penyusunan Hasil 📋
               - Resume kasus
               - Uraian kronologis
               - Analisis yuridis
               - Alat bukti dan barang bukti
               - Kesimpulan dan saran tindak lanjut
            
            4. Standar Pembuktian ✓
               - Minimal 2 alat bukti
               - Keterkaitan antar bukti
               - Kesesuaian dengan unsur pasal
               - Kelengkapan formal
               - Keakuratan materiil\
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
            {Penjabaran tiap unsur delik}
            {Analisis pemenuhan unsur}

            ### Alat Bukti
            {Daftar dan analisis alat bukti}
            {Keterkaitan antar bukti}

            ## Yurisprudensi
            {Putusan-putusan terkait}
            {Analisis kesesuaian}

            ## Rekomendasi
            {Tindak lanjut penyelidikan}
            {Tindak lanjut daftar pemeriksaan}
            {Tindak lanjut koordinasi instansi terkait}

            ## Kesimpulan
            {Rangkuman analisis}
            {Rekomendasi tindak lanjut}

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
