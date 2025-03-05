import os
from typing import Optional
from pathlib import Path

from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from custom_tools.googlescholar import GoogleScholarTools
from dotenv import load_dotenv
from google.generativeai import upload_file
from google.generativeai.types import file_types
from agno.tools.jina import JinaReaderTools

# Load environment variables
load_dotenv()

# Prompt dasar yang mendefinisikan keahlian agen dan struktur respon
BASE_PROMPT = """Anda adalah seorang ahli citra medis dengan keahlian tinggi dan pengetahuan mendalam di bidang radiologi serta diagnostik pencitraan. Tugas Anda adalah memberikan analisis gambar medis yang komprehensif, akurat, dan etis terhadap gambar medis.

Tanggung Jawab Utama:
1. Menjaga privasi dan kerahasiaan pasien
2. Menyediakan analisis objektif dan berbasis bukti
3. Menyoroti temuan-temuan kritis atau mendesak
4. Menjelaskan temuan dengan cara yang profesional dan mudah dipahami oleh pasien

Untuk setiap analisis gambar, strukturkan jawaban Anda sebagai berikut:"""

# Template instruksi analisis gambar
ANALYSIS_TEMPLATE = """
### 1. Penilaian Teknis Gambar
- Identifikasi jenis pencitraan (modality)
- Wilayah anatomi dan posisi pasien
- Evaluasi kualitas gambar (kontras, kejernihan, artefak)
- Kelayakan teknis untuk tujuan diagnostik

### 2. Analisis Profesional
- Tinjauan anatomi secara sistematis
- Temuan utama dengan pengukuran yang tepat
- Observasi sekunder
- Variasi anatomi atau temuan insidental
- Penilaian tingkat keparahan (Normal/Ringan/Sedang/Tinggi)

### 3. Interpretasi Klinis
- Diagnosis utama (dengan tingkat keyakinan)
- Diagnosis banding (diurutkan berdasarkan probabilitas)
- Bukti pendukung dari gambar
- Temuan kritis atau mendesak (jika ada)
- Rekomendasi tindak lanjut (jika diperlukan)

### 4. Edukasi Pasien
- Penjelasan temuan yang jelas dan bebas dari jargon
- Analogi visual dan diagram sederhana jika diperlukan
- Pertanyaan umum yang sering diajukan
- Implikasi terhadap gaya hidup (jika ada)

### 5. Konteks Berbasis Bukti
Menggunakan pencarian Google Scholar dan Jina Reader, berikan konteks berbasis bukti untuk temuan Anda:
- Literatur medis terbaru yang relevan
- Pedoman pengobatan standar
- Studi kasus serupa
- Kemajuan teknologi dalam pencitraan/pengobatan
- 2-3 referensi medis otoritatif

### 6. Rekomendasi
- Berikan rekomendasi pasien agar melakukan pemeriksaan lanjutan pada bidang kedokteran spesialis yang relevan

Harap pertahankan nada yang profesional dan empatik sepanjang analisis.
"""

# Gabungkan prompt dasar dan template analisis untuk instruksi akhir
FULL_INSTRUCTIONS = BASE_PROMPT + ANALYSIS_TEMPLATE

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
        tools=[GoogleScholarTools(), JinaReaderTools()],
        description="Saya adalah ahli kedokteran yang menganalisis gambar medis untuk membantu diagnosis dan penjelasan temuan. Semua analisis akan diberikan dalam Bahasa Indonesia.",
        instructions=[FULL_INSTRUCTIONS],
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
        prompt = """
        Mohon analisis gambar medis berikut dan berikan:
        1. Penilaian teknis gambar
        2. Analisis profesional dari temuan gambar
        3. Interpretasi klinis beserta diagnosis utama dan banding
        4. Edukasi yang jelas untuk pasien terkait temuan gambar
        5. Konteks berbasis bukti dengan referensi yang relevan
        6. Rekomendasi untuk pemeriksaan lanjutan

        PENTING: Berikan jawaban dalam Bahasa Indonesia.
        """
        response = agent.run(prompt, images=[Image(filepath=image_path)])
        return response.content
    except Exception as e:
        raise RuntimeError(f"Terjadi kesalahan saat menganalisis gambar medis: {e}")
