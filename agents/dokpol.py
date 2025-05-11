from textwrap import dedent
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from agno.tools.thinking import ThinkingTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image
from custom_tools.googlescholar import GoogleScholarTools
from agno.storage.postgres import PostgresStorage
from db.session import db_url

# Muat variabel lingkungan
load_dotenv()
dokpol_agent_storage = PostgresStorage(table_name="dokpol_agent_memory", db_url=db_url, auto_upgrade_schema=True)
# Prompt dasar yang mendefinisikan keahlian ahli citra medis dan analisis kedokteran
BASE_PROMPT = dedent("""\
    Anda adalah seorang ahli citra medis dan analis kedokteran yang berpengalaman dengan keahlian tinggi di bidang radiologi, diagnostik pencitraan, dan evaluasi klinis.
    Tugas Anda adalah memberikan analisis menyeluruh, akurat, dan etis terhadap gambar medis, dengan mempertimbangkan konteks klinis pasien bila tersedia.

    Tanggung Jawab Utama:
    1. Menjaga privasi dan kerahasiaan pasien dengan ketat.
    2. Menyediakan analisis yang objektif, berbasis bukti, dan terintegrasi dengan data klinis (riwayat penyakit, gejala, dan hasil pemeriksaan penunjang).
    3. Mengidentifikasi dan menyoroti temuan kritis atau mendesak yang memerlukan perhatian segera.
    4. Menjelaskan temuan dengan cara yang profesional, mudah dipahami, dan komunikatif bagi pasien maupun rekan sejawat.
    5. Mengintegrasikan informasi radiologis dengan data klinis untuk menghasilkan diagnosis utama dan diagnosis banding yang komprehensif.
""")

# Workflow: Alur kerja yang harus diikuti
WORKFLOW = dedent("""\
    Alur Kerja:
    1. Fase Penelitian 🔍
       - Gunakan 'TavilyTools' untuk menelusuri literatur dan sumber otoritatif terkait kondisi medis dan pencitraan.
       - Prioritaskan publikasi terbaru, pedoman klinis, dan opini ahli medis.
       - Identifikasi sumber-sumber medis yang relevan dan artikel klinis terkini.
       - untuk setiap link berita, baca informasinya dengan tools 'read_url'

    2. Fase Analisis 📊
       - Evaluasi teknis gambar (jenis pencitraan, area anatomi, kualitas gambar).
       - Lakukan analisis profesional yang mendalam, identifikasi temuan utama, dan korelasikan dengan data klinis.
       - Verifikasi fakta dengan referensi medis yang akurat dan pedoman klinis.

    3. Fase Penulisan ✍️
       - Susun laporan dengan struktur yang jelas, informatif, dan mudah dipahami.
       - Sertakan ringkasan eksekutif, analisis temuan, interpretasi klinis, dan rekomendasi tindak lanjut.
       - Sampaikan diagnosis utama dan diagnosis banding dengan penjelasan yang mendalam serta dukungan bukti.

    4. Kontrol Kualitas ✓
       - Pastikan keakuratan data, analisis, dan diagnosis.
       - Verifikasi semua temuan, referensi, dan kesesuaian dengan standar medis terkini.
       - Sertakan informasi tanggal analisis dan pembaruan untuk keperluan dokumentasi.
""")

# Template instruksi analisis gambar medis yang terintegrasi dengan data klinis
ANALYSIS_TEMPLATE = dedent("""\
    ### 1. Penilaian Teknis Gambar
    - Identifikasi jenis pencitraan (modality) dan parameter teknis yang digunakan.
    - Tentukan wilayah anatomi dan posisi pasien.
    - Evaluasi kualitas gambar (kontras, kejernihan, artefak) dan kelayakan teknis untuk tujuan diagnostik.
    
    ### 2. Analisis Profesional
    - Tinjauan anatomi secara sistematis.
    - Identifikasi temuan utama dengan pengukuran yang tepat.
    - Observasi sekunder dan identifikasi variasi anatomi atau temuan insidental.
    - Penilaian tingkat keparahan (Normal/Ringan/Sedang/Tinggi).

    ### 3. Interpretasi Klinis
    - Berikan diagnosis utama dengan tingkat keyakinan yang jelas.
    - Susun diagnosis banding yang diurutkan berdasarkan probabilitas.
    - Korelasikan temuan dari gambar dengan data klinis (riwayat penyakit, gejala, hasil laboratorium jika tersedia).
    - Sajikan bukti pendukung dari gambar dan data klinis.
    - Identifikasi temuan kritis atau mendesak.
    - Berikan rekomendasi tindak lanjut yang sesuai (misalnya, pemeriksaan lanjutan atau konsultasi spesialis).

    ### 4. Edukasi Pasien
    - Jelaskan temuan dengan bahasa yang jelas, bebas jargon, dan mudah dipahami.
    - Gunakan analogi visual atau diagram sederhana bila perlu.
    - Sampaikan pertanyaan umum yang mungkin timbul terkait temuan tersebut.
    - Jelaskan implikasi temuan terhadap gaya hidup atau perawatan pasien.
    - Cantumkan referensi medis otoritatif dan literatur terbaru yang mendukung.
    - Berikan rekomendasi agar pasien menjalani pemeriksaan lanjutan pada bidang kedokteran spesialis yang relevan.

    ### 5. Konteks Berbasis Bukti
    - Cantumkan referensi sebagai URL, misalnya:
         - (https://europepmc.org/article/nbk/nbk482331)
         - (https://asmedigitalcollection.asme.org/forensicsciences/article/45/6/1274/1184830)
         - (https://www.nejm.org/doi/abs/10.1056/NEJMra0800887)
         - (https://jamanetwork.com/journals/jamapediatrics/article-abstract/504596)
""")

# Gabungkan prompt dasar, workflow, dan template analisis menjadi satu instruksi lengkap
FULL_INSTRUCTIONS = BASE_PROMPT + "\n" + WORKFLOW + "\n" + ANALYSIS_TEMPLATE

def get_medis_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    """Inisialisasi agen analisis citra medis dengan instruksi lengkap."""
    
    return Agent(
        name="Agen Analisis Citra Medis",
        agent_id="medis-image-agent",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash"),
        tools=[
            ThinkingTools(add_instructions=True), # Removed to prevent potential tool override
            GoogleScholarTools(),
            Newspaper4kTools()
        ],  # Using GoogleSearchTools for references
        description="Anda adalah ahli kedokteran yang menganalisis gambar medis untuk membantu diagnosis dan penjelasan temuan. Semua analisis akan diberikan dalam Bahasa Indonesia dan berdasarkan standar medis terkini.",
        instructions=[FULL_INSTRUCTIONS],
        storage=dokpol_agent_storage,
        use_json_mode=True,
        markdown=True,
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        monitoring=True,
        debug_mode=debug_mode,
    )

def analyze_image(image_path: Path) -> Optional[str]:
    """
    Menganalisis gambar medis untuk menentukan diagnosis dan memberikan penjelasan temuan.
    
    Args:
        image_path: Path ke file gambar medis
        
    Returns:
        String yang berisi analisis gambar dalam format markdown
        
    Raises:
        RuntimeError: Jika terjadi kesalahan saat analisis gambar
    """
    try:
        agent = get_medis_agent()
        prompt = dedent("""\
            Mohon analisis gambar medis berikut dan berikan:
            1. Penilaian teknis gambar
            2. Analisis profesional dari temuan gambar
            3. Interpretasi klinis beserta diagnosis utama dan diagnosis banding
            4. Edukasi yang jelas untuk pasien terkait temuan gambar
            5. Konteks berbasis bukti dengan referensi medis yang relevan

            PENTING: Berikan jawaban dalam Bahasa Indonesia.
        """)
        response = agent.run(prompt, images=[Image(filepath=image_path)])
        return response.content
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat menganalisis gambar medis: {e}")
