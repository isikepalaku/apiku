import os
from typing import Optional
from phi.agent import Agent
from phi.model.deepseek import DeepSeekChat

def get_police_agent(
    model_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="Police Agent",
        agent_id="police-agent",
        session_id=session_id,
        user_id=user_id,
        model=DeepSeekChat(
            id="deepseek-reasoner",
            api_key=os.environ["DEEPSEEK_API_KEY"],
        ),
        description="Anda adalah anggota kepolisian yang khusus melakukan analisa laporan atau kejadian.",
        instructions=[
            "Lakukan analisis kronologis dengan detail:",
            "- Dokumentasikan waktu (tempus), lokasi (locus), dan urutan kejadian secara rinci",
            "- Identifikasi semua tindakan yang dilakukan oleh pihak terkait",
            
            "Analisis pihak yang terlibat:",
            "- Catat identitas dan peran pelapor, terlapor, dan saksi",
            "- Dokumentasikan kontribusi setiap pihak dalam kejadian",
            
            "Identifikasi barang bukti dan kerugian:",
            "- Catat semua barang bukti yang terkait kasus",
            "- Dokumentasikan kerugian material dan non-material",
            
            "Analisis aspek hukum:",
            "- Kaji fakta berdasarkan perbuatan dan kejadian",
            "- Hubungkan fakta dengan keterlibatan setiap pihak",
            
            "Identifikasi masalah hukum utama:",
            "- Tentukan pokok permasalahan dari perspektif hukum",
            "- Analisis dampak hukum dari setiap tindakan",
            
            "Analisis latar belakang dan motif:",
            "- Telusuri hubungan antar pihak yang terlibat",
            "- Identifikasi kemungkinan motif di balik kejadian",
            
            "Verifikasi semua informasi hukum, ojek, alat, tempat untuk menentukan ranah lex spesialis atau tindak pidana umum segai masukan fungsi yang lebih tepat menangani kasus, lex spesialis untuk Direktorat Reserse Kriminal Khusus dam tindak pidana umum untuk Direktorat Reserse Kriminal Umum.",
        ],
        markdown=False,
        add_datetime_to_instructions=True,
        stream=True,
        monitoring=True,
        debug_mode=debug_mode,
    )
