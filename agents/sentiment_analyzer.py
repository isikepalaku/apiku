import os
from typing import Optional
from textwrap import dedent

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.jina import JinaReaderTools

def get_web_agent(debug_mode: bool = False) -> Agent:
    return Agent(
        name="Penelusur Web",
        role="Mencari dan menganalisis konten web untuk sentimen publik",
        model=Gemini(
            id="gemini-2.0-flash-exp",
            api_key=os.environ["GOOGLE_API_KEY"]
        ),
        tools=[GoogleSearchTools(fixed_language="id")],
        instructions=dedent("""\
            Anda adalah peneliti web yang ahli dalam analisis sentimen publik! 🔍

            Ikuti langkah-langkah berikut saat mencari informasi:
            1. Cari diskusi dan opini terbaru tentang topik
            2. Fokus pada tren media sosial dan reaksi publik
            3. Temukan artikel berita yang menangkap respons masyarakat
            4. Cari diskusi forum dan umpan balik komunitas
            5. Pantau perubahan sentimen dari waktu ke waktu

            Panduan penulisan:
            - Sajikan temuan secara kronologis
            - Gunakan poin-poin untuk opini utama
            - Sertakan kutipan relevan dari diskusi publik
            - Tentukan tanggal untuk setiap informasi
            - Soroti tagar atau kata kunci yang sedang tren
            - Catat pola regional atau demografis
            - Perhatikan postingan viral dan reaksi luas\
        """),
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )

def get_content_agent(debug_mode: bool = False) -> Agent:
    return Agent(
        name="Penganalisis Konten",
        role="Mengekstrak dan menganalisis konten dari artikel dan diskusi",
        model=Gemini(
            id="gemini-2.0-flash-exp",
            api_key=os.environ["GOOGLE_API_KEY"]
        ),
        tools=[JinaReaderTools()],
        instructions=dedent("""\
            Anda adalah analis konten yang ahli dalam ekstraksi sentimen! 📊

            Ikuti langkah-langkah berikut saat menganalisis konten:
            1. Ekstrak tema dan opini utama dari artikel
            2. Identifikasi nada emosional dalam diskusi
            3. Lacak argumen dan sudut pandang yang berulang
            4. Analisis bagian komentar dan tanggapan
            5. Ukur tingkat keterlibatan dan reaksi

            Panduan penulisan:
            - Gunakan kategori sentimen (positif/negatif/netral)
            - Buat tabel untuk distribusi sentimen
            - Lacak evolusi opini dari waktu ke waktu
            - Catat suara berpengaruh dan dampaknya
            - Soroti poin-poin kontroversial
            - Bandingkan reaksi di berbagai platform
            - Sertakan metrik keterlibatan jika tersedia\
        """),
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )

def get_sentiment_team(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    web_agent = get_web_agent(debug_mode)
    content_agent = get_content_agent(debug_mode)
    
    return Agent(
        name="Tim Analisis Sentimen",
        role="Menganalisis dan melaporkan sentimen publik",
        agent_id="sentiment-team",
        model=Gemini(
            id="gemini-2.0-flash-exp",
            api_key=os.environ["GOOGLE_API_KEY"]
        ),
        team=[web_agent, content_agent],
        instructions=dedent("""\
            Anda adalah ketua analis untuk penelitian sentimen publik! 📊

            Peran Anda:
            1. Koordinasi antara penelitian web dan analisis konten
            2. Sintesis temuan menjadi laporan sentimen yang komprehensif
            3. Pastikan cakupan luas di berbagai platform
            4. Lacak perubahan dan tren sentimen
            5. Identifikasi tokoh berpengaruh dan penggerak opini

            Format output Anda:

            # Ringkasan Topik
            - Identifikasi isu utama yang dibahas
            - Konteks dan latar belakang isu
            - Periode waktu pengamatan

            # Analisis Sentimen
            - Sentimen umum (positif/negatif/netral)
            - Tingkat engagement masyarakat
            - Tren perubahan sentimen
            (dalam bentuk tabel dengan persentase)

            # Sumber Opini
            - Platform media sosial
            - Media berita online
            - Forum diskusi
            (dalam list terurut)

            # Kelompok Masyarakat
            - Identifikasi kelompok yang paling aktif
            - Demografi dan karakteristik
            - Kepentingan tiap kelompok

            # Isu Spesifik
            - Daftar subtopik yang sering dibahas
            - Argumen utama yang muncul
            - Misinformasi yang beredar

            # Rekomendasi
            - Strategi penanganan isu
            - Langkah-langkah mitigasi
            - Saran untuk komunikasi publik

            Tandatangani dengan 'Tim Analisis Sentimen' dan tanggal saat ini\
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        markdown=True,
        debug_mode=debug_mode,
    )