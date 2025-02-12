import json
import os
from typing import Iterator, Optional, List

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.storage.workflow.postgres import PostgresWorkflowStorage
from agno.tools.googlesearch import GoogleSearchTools
from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field

from workflows.settings import workflow_settings
from db.session import db_url

class ModusOperandi(BaseModel):
    metode: str = Field(..., description="Metode atau cara yang digunakan dalam kejahatan")
    deskripsi: str = Field(..., description="Penjelasan detail tentang modus operandi")
    contoh_kasus: List[str] = Field(..., description="Contoh-contoh kasus yang menggunakan modus ini")
    target_sasaran: str = Field(..., description="Target atau sasaran yang sering menjadi korban")

class AnalisaPola(BaseModel):
    kategori_kejahatan: str = Field(..., description="Kategori kejahatan yang sedang dianalisis")
    deskripsi_umum: str = Field(..., description="Deskripsi umum tentang kategori kejahatan ini")
    modus_operandi: List[ModusOperandi] = Field(..., description="Daftar modus operandi yang teridentifikasi")
    lokasi_rawan: str = Field(..., description="Identifikasi daerah rawan kejahatan")
    waktu_rawan: str = Field(..., description="Waktu-waktu rawan terjadinya kejahatan")

class AnalisaTren(BaseModel):
    tren_modus: str = Field(..., description="Tren perubahan modus operandi terkini")
    faktor_pendorong: str = Field(..., description="Faktor-faktor yang mendorong perubahan modus")
    pola_musiman: str = Field(..., description="Pola musiman dalam penggunaan modus tertentu")
    prediksi_perkembangan: str = Field(..., description="Prediksi perkembangan modus ke depan")

class AnalisatorTrenKejahatan(Workflow):
    agen_analisa_modus: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[GoogleSearchTools()],
        instructions=[
            "Analisis pola kejahatan berdasarkan kategori yang diberikan.",
            "Gunakan Google Search untuk menemukan contoh kasus nyata.",
            "Jangan memberikan komentar subjektif atau penilaian.",
            "Fokus pada identifikasi pola dan karakteristik modus operandi.",
            "Berikan response dalam format JSON yang sesuai dengan struktur berikut:",
            "{",
            "  'kategori_kejahatan': 'string',",
            "  'deskripsi_umum': 'string',",
            "  'modus_operandi': [{",
            "    'metode': 'string',",
            "    'deskripsi': 'string',",
            "    'contoh_kasus': ['string'],",
            "    'target_sasaran': 'string'",
            "  }],",
            "  'lokasi_rawan': 'string',",
            "  'waktu_rawan': 'string'",
            "}"
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=AnalisaPola,
        structured_outputs=True,
        debug_mode=False,
    )

    agen_analisa_tren: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash-lite-preview-02-05"),
        tools=[GoogleSearchTools()],
        instructions=[
            "Analisis tren dan pola perubahan modus operandi kejahatan.",
            "Gunakan Google Search untuk mendapatkan data terkini.",
            "Fokus pada fakta dan pola yang teridentifikasi.",
            "PENTING: Response HARUS dalam format JSON dengan SEMUA field berikut:",
            "{",
            "  'tren_modus': 'Deskripsi lengkap tren perubahan modus operandi',",
            "  'faktor_pendorong': 'Faktor-faktor yang mempengaruhi perubahan modus (WAJIB diisi)',",
            "  'pola_musiman': 'Pola waktu/musim yang terdeteksi (WAJIB diisi)',",
            "  'prediksi_perkembangan': 'Analisis kemungkinan perkembangan (WAJIB diisi)'",
            "}",
            "",
            "Pastikan SEMUA field terisi dengan informasi yang relevan.",
            "Jika tidak ada data spesifik, isi dengan 'Belum teridentifikasi'.",
            "JANGAN mengosongkan atau menghilangkan field apapun.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=AnalisaTren,
        structured_outputs=True,
        debug_mode=False,
    )

    agen_rekomendasi: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash-lite-preview-02-05"),
        tools=[GoogleSearchTools()],
        instructions=[
            "Analisis pola kejahatan dan identifikasi karakteristik khusus:",
            "1. Cari kasus-kasus serupa menggunakan Google Search",
            "2. Identifikasi pola kemunculan kasus sejenis",
            "3. Analisis karakteristik khusus dari setiap modus",
            "4. Jelaskan indikator awal terjadinya kejadian serupa",
            "5. Tentukan pola geografis dan temporal",
            "Fokus pada pola dan karakteristik, bukan pada saran pencegahan.",
            "Berikan data faktual dan contoh kasus nyata.",
            "Hindari memberikan komentar subjektif atau penilaian.",
            "Format output dalam bentuk analisis terstruktur."
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=False,
    )

    agen_laporan: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        instructions=[
            "Buat laporan analisis kejahatan yang objektif:",
            "1. Identifikasi Modus:",
            "   - Karakteristik utama",
            "   - Pola yang teridentifikasi",
            "   - Contoh kasus nyata",
            "",
            "2. Detail Kejadian:",
            "   - Pola waktu dan lokasi",
            "   - Karakteristik target",
            "   - Pola operasional",
            "",
            "3. Analisis Perkembangan:",
            "   - Perubahan pola",
            "   - Variasi modus",
            "   - Area penyebaran",
            "",
            "4. Data Pendukung:",
            "   - Statistik kejadian",
            "   - Pola geografis",
            "   - Tren temporal",
            "",
            "Format Penting:",
            "- Gunakan poin-poin dan paragraf",
            "- Fokus pada data dan fakta",
            "- Sertakan contoh kasus konkret",
            "- Hindari memberikan komentar subjektif",
            "- Jangan tambahkan penilaian pribadi",
            "- Jangan berikan respon dalam bentuk tabel"
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=False,
    )

    def _create_default_analisa_tren(self, kategori_kejahatan: str) -> AnalisaTren:
        """Create a default AnalisaTren object when analysis fails."""
        return AnalisaTren(
            tren_modus=f"Analisis tren untuk {kategori_kejahatan} belum dapat dilakukan",
            faktor_pendorong="Data faktor pendorong belum teridentifikasi",
            pola_musiman="Pola musiman belum dapat diidentifikasi",
            prediksi_perkembangan="Prediksi perkembangan belum dapat dilakukan"
        )

    def get_analisa_modus(self, kategori_kejahatan: str) -> Optional[AnalisaPola]:
        try:
            logger.info(f"Menganalisis modus operandi untuk: {kategori_kejahatan}")
            
            formatted_input = f"""
Lakukan analisis kejahatan untuk kategori: {kategori_kejahatan}

Gunakan Google Search untuk menemukan contoh-contoh kasus nyata.
Berikan hasil analisis dalam format JSON berikut:
{{
    "kategori_kejahatan": "{kategori_kejahatan}",
    "deskripsi_umum": "Penjelasan objektif tentang kategori kejahatan",
    "modus_operandi": [
        {{
            "metode": "Nama/jenis modus operandi",
            "deskripsi": "Penjelasan teknis tentang cara kerja",
            "contoh_kasus": ["Contoh kasus 1", "Contoh kasus 2"],
            "target_sasaran": "Karakteristik target/korban"
        }}
    ],
    "lokasi_rawan": "Area yang sering terjadi kasus",
    "waktu_rawan": "Pola waktu kejadian"
}}

PENTING: Berikan data faktual, hindari komentar subjektif.
"""
            response: RunResponse = self.agen_analisa_modus.run(formatted_input)

            if not response or not response.content:
                logger.warning("Response kosong")
                return None

            if isinstance(response.content, AnalisaPola):
                return response.content

            if isinstance(response.content, str):
                try:
                    # Extract JSON if embedded in narrative text
                    content = response.content
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        parsed_content = json.loads(json_str)
                        return AnalisaPola(**parsed_content)
                    else:
                        logger.error("Tidak ditemukan konten JSON")
                        logger.debug(f"Response content: {content}")
                except Exception as parse_error:
                    logger.error(f"Gagal parsing response: {parse_error}")
                    logger.debug(f"Response content: {response.content}")

            return None

        except Exception as e:
            logger.error(f"Error dalam get_analisa_modus: {str(e)}")
            logger.debug("Detail error:", exc_info=True)
            return None

    def get_analisa_tren(
        self, kategori_kejahatan: str, analisa_pola: AnalisaPola
    ) -> Optional[AnalisaTren]:
        try:
            agent_input = {
                "kategori_kejahatan": kategori_kejahatan,
                **(analisa_pola.model_dump() if isinstance(analisa_pola, AnalisaPola) else {"analisa_pola": str(analisa_pola)})
            }

            formatted_input = f"""
Analisis tren dan perkembangan untuk kategori: {kategori_kejahatan}

Data analisa sebelumnya:
{json.dumps(agent_input, indent=2, ensure_ascii=False)}

Hasilkan analisis LENGKAP dalam format JSON berikut (SEMUA field WAJIB diisi):
{{
    "tren_modus": "Deskripsi lengkap perubahan pola yang terdeteksi",
    "faktor_pendorong": "Daftar dan penjelasan faktor-faktor yang teridentifikasi",
    "pola_musiman": "Analisis pola berdasarkan waktu/musim",
    "prediksi_perkembangan": "Analisis kemungkinan perkembangan ke depan"
}}

PENTING:
- SEMUA field di atas WAJIB diisi
- Jika tidak ada data spesifik, isi dengan "Belum teridentifikasi"
- JANGAN mengosongkan atau menghilangkan field apapun
- Gunakan Google Search untuk mendapatkan data aktual
- Fokus pada data dan pola, hindari spekulasi
"""
            response: RunResponse = self.agen_analisa_tren.run(formatted_input)

            if not response or not response.content:
                logger.warning("Response kosong dari agent")
                return self._create_default_analisa_tren(kategori_kejahatan)

            if isinstance(response.content, AnalisaTren):
                return response.content

            if isinstance(response.content, str):
                try:
                    content = response.content
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        parsed_content = json.loads(json_str)
                        # Ensure all required fields exist
                        required_fields = ['tren_modus', 'faktor_pendorong', 'pola_musiman', 'prediksi_perkembangan']
                        for field in required_fields:
                            if field not in parsed_content:
                                parsed_content[field] = "Belum teridentifikasi"
                        return AnalisaTren(**parsed_content)
                    else:
                        logger.error("Tidak ditemukan konten JSON dalam response")
                        return self._create_default_analisa_tren(kategori_kejahatan)
                except Exception as parse_error:
                    logger.error(f"Gagal parsing tren: {parse_error}")
                    logger.debug(f"Response content: {response.content}")
                    return self._create_default_analisa_tren(kategori_kejahatan)

            return self._create_default_analisa_tren(kategori_kejahatan)

        except Exception as e:
            logger.error(f"Error dalam get_analisa_tren: {str(e)}")
            logger.debug("Detail error:", exc_info=True)
            return self._create_default_analisa_tren(kategori_kejahatan)

    def get_rekomendasi(self, kategori_kejahatan: str, analisa_tren: AnalisaTren) -> Optional[str]:
        try:
            agent_input = {
                "kategori_kejahatan": kategori_kejahatan,
                **(analisa_tren.model_dump() if isinstance(analisa_tren, AnalisaTren) else {"analisa_tren": str(analisa_tren)})
            }

            formatted_input = f"""
Analisis karakteristik dan pola untuk: {kategori_kejahatan}

Data analisis tren:
{json.dumps(agent_input, indent=2, ensure_ascii=False)}

Berikan analisis yang mencakup:
1. Pola dan karakteristik yang teridentifikasi
2. Contoh kasus serupa dari berbagai daerah
3. Indikator awal kejadian sejenis
4. Variasi modus yang mungkin berkembang

Gunakan Google Search untuk menemukan contoh kasus aktual.
Fokus pada identifikasi pola dan karakteristik, bukan pencegahan.
"""
            response: RunResponse = self.agen_rekomendasi.run(formatted_input)

            if not response or not response.content:
                return None

            content = response.content.strip() if isinstance(response.content, str) else str(response.content)
            return content if content else None

        except Exception as e:
            logger.error(f"Error dalam get_rekomendasi: {str(e)}")
            logger.debug("Detail error:", exc_info=True)
            return None

    def run(self, kategori_kejahatan: str) -> Iterator[RunResponse]:
        logger.info(f"Memulai analisis untuk kategori: {kategori_kejahatan}")

        # Step 1: Analisis Modus Operandi
        analisa_pola: Optional[AnalisaPola] = self.get_analisa_modus(kategori_kejahatan)
        if analisa_pola is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Gagal menganalisis modus operandi untuk: {kategori_kejahatan}",
            )
            return

        # Step 2: Analisis Tren
        analisa_tren: Optional[AnalisaTren] = self.get_analisa_tren(kategori_kejahatan, analisa_pola)
        if analisa_tren is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Gagal menganalisis tren kejahatan",
            )
            return

        # Step 3: Analisis Pattern
        rekomendasi: Optional[str] = self.get_rekomendasi(kategori_kejahatan, analisa_tren)

        try:
            logger.info("Menyusun laporan final")
            report_input = {
                "kategori_kejahatan": kategori_kejahatan,
                **(analisa_pola.model_dump() if isinstance(analisa_pola, AnalisaPola) else {"analisa_pola": str(analisa_pola)}),
                **(analisa_tren.model_dump() if isinstance(analisa_tren, AnalisaTren) else {"analisa_tren": str(analisa_tren)}),
                "rekomendasi": rekomendasi or "Tidak ada analisis tambahan",
            }
            
            logger.debug(f"Input laporan: {json.dumps(report_input, indent=2, ensure_ascii=False)}")
            
            def format_modus(modus, index):
                """Format satu entri modus operandi"""
                contoh_kasus = os.linesep.join(f"  * {kasus}" for kasus in modus['contoh_kasus'])
                return f"""MODUS #{index + 1}: {modus['metode']}
--------------------
• Pola Operasi:
  {modus['deskripsi']}

• Kasus Serupa:
{contoh_kasus}

• Karakteristik Target:
  {modus['target_sasaran']}"""

            # Format semua modus operandi
            modus_sections = []
            for i, modus in enumerate(report_input.get('modus_operandi', [])):
                modus_section = format_modus(modus, i)
                modus_sections.append(modus_section)
            
            modus_content = os.linesep + os.linesep.join(modus_sections) + os.linesep if modus_sections else "Belum ada modus yang teridentifikasi"

            formatted_report = f"""ANALISIS POLA KEJAHATAN
=====================

1. IDENTIFIKASI UMUM
------------------
• Kategori: {report_input['kategori_kejahatan']}
• Deskripsi:
  {report_input.get('deskripsi_umum', 'Tidak ada deskripsi')}

2. POLA MODUS OPERANDI
-------------------{modus_content}

• Pola Lokasi:
  {report_input.get('lokasi_rawan', 'Belum teridentifikasi')}

• Pola Waktu:
  {report_input.get('waktu_rawan', 'Belum teridentifikasi')}

3. PERKEMBANGAN TREN
-----------------
• Tren Saat Ini:
  {report_input.get('tren_modus', 'Belum ada data tren')}

• Faktor Pendorong:
  {report_input.get('faktor_pendorong', 'Belum teridentifikasi')}

• Pola Musiman:
  {report_input.get('pola_musiman', 'Belum teridentifikasi')}

• Proyeksi:
  {report_input.get('prediksi_perkembangan', 'Belum ada proyeksi')}

4. ANALISIS LANJUTAN
-----------------
{report_input.get('rekomendasi', 'Tidak ada analisis tambahan')}
"""
            logger.debug(f"Format laporan final: {formatted_report}")
            final_response: RunResponse = self.agen_laporan.run(formatted_report)

            if not final_response or not final_response.content:
                yield RunResponse(
                    content="Gagal membuat laporan akhir",
                    event=RunEvent.workflow_completed,
                )
                return

            yield RunResponse(content=final_response.content, event=RunEvent.workflow_completed)

        except Exception as e:
            logger.error(f"Error dalam pembuatan laporan: {str(e)}")
            logger.debug("Detail error:", exc_info=True)
            yield RunResponse(
                content=f"Gagal membuat laporan: {str(e)}",
                event=RunEvent.workflow_completed,
            )

def get_analisator_tren_kejahatan(debug_mode: bool = False) -> AnalisatorTrenKejahatan:
    """Create and configure the workflow instance."""
    workflow = AnalisatorTrenKejahatan(
        workflow_id="analisis-modus-kejahatan",
        storage=PostgresWorkflowStorage(
            table_name="analisa_modus_kejahatan_workflows",
            db_url=db_url,
        ),
        debug_mode=debug_mode,
    )

    if debug_mode:
        logger.info("Mode debug aktif untuk semua agen")
        workflow.agen_analisa_modus.debug_mode = True
        workflow.agen_analisa_tren.debug_mode = True
        workflow.agen_rekomendasi.debug_mode = True
        workflow.agen_laporan.debug_mode = True

    return workflow
