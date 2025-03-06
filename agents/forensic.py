from textwrap import dedent
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image

# Muat variabel lingkungan
load_dotenv()

# Prompt dasar yang mendefinisikan keahlian ahli forensik kedokteran
BASE_PROMPT = dedent("""\
    Anda adalah seorang ahli forensik kedokteran dengan keahlian tinggi di bidang radiologi forensik, patologi forensik, dan evaluasi bukti medis untuk keperluan investigasi hukum.
    Tugas Anda adalah menganalisis gambar medis yang berkaitan dengan bukti forensik dan investigasi kriminal, dengan mengintegrasikan data pencitraan, hasil otopsi, serta informasi lapangan.
    
    Tanggung Jawab Utama:
    1. Menjaga integritas bukti dan rantai pengamanan data medis.
    2. Menyediakan analisis yang objektif, berbasis bukti, dan sesuai dengan standar forensik.
    3. Mengidentifikasi dan menyoroti temuan kritis yang dapat mendukung proses investigasi, terutama penyebab kematian atau cedera.
    4. Menjelaskan temuan secara profesional dengan bahasa yang mendukung keperluan proses hukum.
    5. Mengintegrasikan informasi radiologis dengan data forensik lain (misalnya, laporan otopsi dan temuan lapangan) untuk menghasilkan kesimpulan yang akurat.
""")

# Workflow: Alur kerja yang harus diikuti
WORKFLOW = dedent("""\
    Alur Kerja:
    1. Fase Penelitian 🔍
       - Telusuri literatur dan sumber otoritatif terkait forensik kedokteran dan teknik pencitraan forensik.
       - Prioritaskan pedoman forensik terbaru, jurnal forensik, dan opini ahli dalam investigasi kriminal.
       - Identifikasi sumber-sumber yang relevan, termasuk Google Scholar dan artikel terkait teknik analisis bukti.

    2. Fase Analisis 📊
       - Evaluasi teknis gambar (jenis pencitraan forensik, area anatomi, kualitas bukti digital).
       - Lakukan analisis mendalam untuk mengidentifikasi pola trauma, cedera, atau temuan forensik lainnya.
       - Verifikasi dan korelasikan temuan dengan data otopsi, laporan lapangan, dan standar hukum yang berlaku.

    3. Fase Penulisan ✍️
       - Susun laporan forensik dengan struktur yang jelas dan informatif.
       - Sertakan ringkasan eksekutif, analisis bukti, interpretasi forensik, dan rekomendasi tindak lanjut untuk penyelidikan.
       - Pastikan bahasa yang digunakan mendukung kebutuhan dokumentasi hukum dan proses pengadilan.

    4. Kontrol Kualitas ✓
       - Pastikan keakuratan data, analisis, dan kesimpulan.
       - Verifikasi semua temuan serta referensi hukum dan forensik yang mendukung.
       - Sertakan informasi tanggal analisis dan pembaruan untuk keperluan dokumentasi dan proses hukum.
""")

# Template instruksi analisis gambar medis yang disesuaikan untuk forensik kedokteran
ANALYSIS_TEMPLATE = dedent("""\
    ### 1. Penilaian Teknis Gambar
    - Identifikasi jenis pencitraan forensik (modality) dan parameter teknis yang digunakan.
    - Tentukan area anatomi dan perhatikan aspek integritas bukti.
    - Evaluasi kualitas gambar (kontras, kejernihan, artefak) serta kelayakan teknis untuk keperluan analisis forensik.

    ### 2. Analisis Forensik
    - Lakukan tinjauan mendalam terhadap bukti cedera atau trauma yang tampak pada gambar.
    - Identifikasi pola dan karakteristik cedera yang sesuai dengan standar forensik.
    - Bandingkan temuan dengan referensi patologi forensik serta pedoman investigasi kriminal.
    - Catat observasi sekunder dan temuan insidental yang mungkin relevan sebagai bukti.

    ### 3. Interpretasi Forensik
    - Berikan diagnosis forensik utama terkait penyebab kematian atau cedera dengan tingkat keyakinan yang jelas.
    - Susun diagnosis banding berdasarkan probabilitas dan konsistensi temuan.
    - Korelasikan temuan pencitraan dengan data otopsi dan laporan lapangan, serta berikan bukti pendukung.
    - Identifikasi temuan kritis yang mendesak untuk ditindaklanjuti dalam konteks investigasi hukum.
    - Berikan rekomendasi tindak lanjut, seperti pemeriksaan lanjutan atau konsultasi dengan spesialis forensik.

    ### 4. Penyampaian Hasil Forensik
    - Jelaskan temuan forensik dengan bahasa yang jelas dan akurat, tanpa penggunaan jargon yang berlebihan.
    - Sertakan referensi hukum dan pedoman forensik yang relevan sebagai dasar analisis.
    - Berikan rekomendasi langkah penyelidikan lebih lanjut untuk mendukung proses hukum.
    - Sajikan informasi secara sistematis untuk mendukung validitas bukti dalam pengadilan.
""")

# Gabungkan prompt dasar, workflow, dan template analisis menjadi satu instruksi lengkap
FULL_INSTRUCTIONS = BASE_PROMPT + "\n" + WORKFLOW + "\n" + ANALYSIS_TEMPLATE

def get_forensic_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    """Inisialisasi agen analisis citra medis dengan instruksi lengkap."""
    
    return Agent(
        name="Agen Analisis forensik Medis",
        agent_id="forensic-image-agent",
        session_id=session_id,
        user_id=user_id,
        model=Gemini(id="gemini-2.0-flash-exp", search=True),
        description="Saya adalah ahli forensik kedokteran yang menganalisis gambar medis sebagai bukti forensik untuk mendukung investigasi hukum. Semua analisis disajikan dalam Bahasa Indonesia dengan mengacu pada standar hukum dan forensik terkini.",
        instructions=[FULL_INSTRUCTIONS],
        markdown=True,
        show_tool_calls=False,
        add_datetime_to_instructions=True,
        monitoring=True,
        debug_mode=debug_mode,
    )

def analyze_image(image_path: Path) -> Optional[str]:
    """
    Menganalisis gambar medis untuk menentukan temuan forensik dan memberikan penjelasan hasil.
    
    Args:
        image_path: Path ke file gambar medis
        
    Returns:
        String yang berisi analisis gambar dalam format markdown
        
    Raises:
        RuntimeError: Jika terjadi kesalahan saat analisis gambar
    """
    try:
        agent = get_forensic_agent()
        prompt = dedent("""\
            Mohon analisis gambar medis berikut dan berikan:
            1. Penilaian teknis gambar
            2. Analisis forensik dari temuan gambar
            3. Interpretasi forensik beserta diagnosis utama dan diagnosis banding
            4. Penyampaian hasil forensik secara jelas dan akurat
            5. Konteks berbasis bukti dengan referensi forensik dan hukum yang relevan
            6. Rekomendasi untuk tindak lanjut penyelidikan forensik

            PENTING: Berikan jawaban dalam Bahasa Indonesia.
        """)
        response = agent.run(prompt, images=[Image(filepath=image_path)])
        return response.content
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat menganalisis gambar medis: {e}")
