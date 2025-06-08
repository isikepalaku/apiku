from datetime import datetime
from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.storage.postgres import PostgresStorage
from db.session import db_url
from typing import Optional

fidusia_agent_storage = PostgresStorage(table_name="fidusia_agent", db_url=db_url, auto_upgrade_schema=True)

# Fungsi untuk inisialisasi agen penyidik tipikor
def get_corruption_investigator(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        agent_id="penyidik-tipikor",
        session_id=session_id,
        user_id=user_id,
        name="Penyidik Tipikor",
        role="Penyidik khusus tindak pidana korupsi",
        model=Gemini(id="gemini-2.5-pro-preview-05-06"),
        tools=[DuckDuckGoTools(), Newspaper4kTools()],
        use_json_mode=True,
        description=dedent("""Anda adalah asisten penyidik kepolisian yang ahli dalam penanganan kasus tindak pidana korupsi Indonesia."""),
        instructions=dedent("""\
        **Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.
        
            1. Metodologi Penelitian Hukum 🔍
               - Lakukan pencarian web kasus terkait
               - Fokus pada putusan pengadilan terkait
               - Prioritaskan yurisprudensi terbaru
               - Identifikasi pasal-pasal kunci dan penerapannya
               - Telusuri pola modus operandi dari kasus serupa

            2. Kerangka Analisis 📊
               - ekstrak URL web temuan
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
               - Buat dalam tabel jika memungkinkan

            4. Standar Pembuktian ✓
               - Pastikan akurasi kutipan pasal
               - Jaga ketepatan analisis hukum
               - Sajikan perspektif berimbang
               - Dukung dengan yurisprudensi
               - Lengkapi dengan analisis

            5. Berikan daftar link sumber yang ditemukan
            6. Lakukan anailisis dengan sangat mendetil dan berikan penjelasan dengan terstruktur dan jelas
            7. Ingat!!! setelah proses dilakukan berikan hasil, jangan mengulang proses yang sama
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
            {Daftar dan analisis alat bukti dalam tabel jika memungkinkan}

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
        storage=fidusia_agent_storage,
        debug_mode=debug_mode,
    )
