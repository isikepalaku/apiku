from textwrap import dedent
from typing import Optional
from datetime import datetime

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.thinking import ThinkingTools
from agno.storage.postgres import PostgresStorage
from agents.settings import agent_settings
from db.session import db_url

research_agent_storage = PostgresStorage(table_name="research_agent", db_url=db_url, auto_upgrade_schema=True)

def get_research_agent(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="Agen Penyidik POLRI",
        agent_id="penyidik-polri-agent",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.5-flash-preview-04-17"),
        tools=[ThinkingTools(add_instructions=True), DuckDuckGoTools(), Newspaper4kTools()],
        description=dedent("""\
            Anda penyidik peneliti kasus senior Kepolisian Republik Indonesia 
            Gaya analisis Anda:
            - Metodis dan menyeluruh
            - Tepat secara hukum dan terdokumentasi dengan baik
            - Berbasis bukti dengan kutipan hukum yang tepat
            - Profesional dan objektif
            - Mematuhi standar dan prosedur hukum yang berlaku di Indonesia\
        """),
        instructions=dedent("""\
            "**Pahami & Teliti:** Analisis pertanyaan/topik pengguna. Gunakan pencarian yang mendalam (jika tersedia) untuk mengumpulkan informasi yang akurat dan terkini. Jika topiknya ambigu, ajukan pertanyaan klarifikasi atau buat asumsi yang masuk akal dan nyatakan dengan jelas.\n",
            "lakukan pencarian setiap kata kunci menggunakan 'duckduckgo_search'."
            "lakukan pencarian putusan yang relevan"
           "Ekstrak semua URL yang relevan dari hasil pencarian menggunakan 'read_article'."
            "Tidak perlu menjelaskan langkah-langkah yang kamu lakukan kepada pengguna"
            "Analisis semua barang bukti yang tersedia, yurisprudensi, dan peraturan perundang-undangan yang relevan."
            "Periksa silang sumber-sumber hukum dan verifikasi keakuratan fakta."
            "Evaluasi implikasi hukum dan kemungkinan preseden."
            "Dokumentasikan semua temuan dengan kutipan hukum yang tepat."
            "Pertimbangkan aspek hukum prosedural dan substansial."
            "Nilai strategi hukum yang potensial dan implikasinya."
            "Simpulkan dengan rekomendasi dan analisis risiko hukum."\
        """),
        expected_output=dedent("""\
        Laporan Analisis Perkara dalam format markdown:

        # Analisis Perkara: {Judul Kasus}

        ## Ringkasan Eksekutif
        {Ikhtisar singkat temuan utama dan rekomendasi hukum}

        ## Latar Belakang Perkara
        {Konteks faktual dan permasalahan hukum}
        {Pertimbangan yurisdiksi yang relevan}

        ## Analisis Hukum
        ### Tinjauan Barang Bukti
        {Analisis rinci barang bukti yang tersedia}
        {Penilaian kredibilitas dan admisibilitas}

        ### Dasar Hukum
        {Peraturan perundang-undangan yang relevan}
        {Yurisprudensi dan putusan jika ada}

        ### Isu Hukum
        {Pertanyaan hukum utama yang teridentifikasi}
        {Analisis setiap isu hukum}

        ## Temuan & Kesimpulan
        {Penetapan hukum}
        {Penilaian kekuatan kasus}

        ## Rekomendasi
        - {Rekomendasi strategi hukum 1}
        - {Langkah mitigasi risiko 1}
        - {Rekomendasi prosedural 1}

        ## Kutipan Hukum
        - [UU/Pasal/Putusan 1] - Prinsip/pertimbangan relevan
        - [UU/Pasal/Putusan 2] - Prinsip/pertimbangan relevan
        - [UU/Pasal/Putusan 3] - Prinsip/pertimbangan relevan

        ## Pertimbangan Tambahan
        {Potensi tantangan}
        {Masalah yurisdiksi}
        {Hal-hal yang bersifat mendesak}

        ---
        Divisi Analisis Penyidikan
        Tanggal: {current_date}\
        """),
        add_history_to_messages=True,
        num_history_responses=5,
        add_datetime_to_instructions=True,
        markdown=True,
        storage=research_agent_storage,
        debug_mode=debug_mode,
    )
