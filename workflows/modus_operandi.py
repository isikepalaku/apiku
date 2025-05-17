import json
from typing import Iterator, Optional, List

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.redis import RedisStorage
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field

from workflows.settings import workflow_settings

# Define Redis storage instance
redis_storage = RedisStorage(
    prefix="modus_operandi",
    host="agno-demo-app-dev-redis",  # Docker container name for Redis
    port=6379,
)

class ModusOperandiKejahatan(BaseModel):
    metode: str = Field(..., description="Metode operasi yang digunakan pelaku kejahatan")
    deskripsi: str = Field(..., description="Deskripsi detail modus operandi dan teknik yang digunakan")
    contoh_kasus: List[str] = Field(..., description="Dokumentasi kasus-kasus yang menggunakan modus serupa")
    target_sasaran: str = Field(..., description="Profil dan karakteristik target/korban")
    tingkat_ancaman: str = Field(..., description="Tingkat bahaya/ancaman dari modus ini")
    bukti_pendukung: List[str] = Field(..., description="Bukti-bukti yang sering ditemukan")

class AnalisaPolisional(BaseModel):
    kategori_kejahatan: str = Field(..., description="Klasifikasi jenis kejahatan")
    deskripsi_umum: str = Field(..., description="Deskripsi umum pola kejahatan")
    modus_operandi: List[ModusOperandiKejahatan] = Field(..., description="Daftar modus operandi teridentifikasi")
    lokasi_rawan: str = Field(..., description="Area dan lokasi rawan kejahatan")
    waktu_rawan: str = Field(..., description="Pola waktu kejadian")
    profil_pelaku: str = Field(..., description="Karakteristik umum pelaku")

class AnalisaTrenKejahatan(BaseModel):
    tren_modus: str = Field(..., description="Analisis tren perubahan modus operandi")
    faktor_pendorong: str = Field(..., description="Faktor-faktor penyebab dan pendorong")
    pola_musiman: str = Field(..., description="Pola berdasarkan waktu/musim")
    prediksi_perkembangan: str = Field(..., description="Proyeksi perkembangan modus")
    rekomendasi_tindakan: str = Field(..., description="Rekomendasi tindakan preventif")

class SistemAnalisisIntelijen(Workflow):
    agen_analisis_modus: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        tools=[TavilyTools(), Newspaper4kTools()],
        instructions=[
            "Lakukan pencarian kata kunci untuk kasus yang pernah terjadi.",
            "Gunakan data kasus nyata sebagai referensi lampirkan link.",
            "Evaluasi tingkat ancaman dan pola operasional.",
            "Identifikasi bukti-bukti yang sering ditemukan.",
            "Tentukan karakteristik pelaku dan korban.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=AnalisaPolisional,
        structured_outputs=True,
        debug_mode=False,
    )

    agen_analisis_tren: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        instructions=[
            "Analisis tren dan perubahan modus operandi kejahatan.",
            "Identifikasi faktor pendorong dan pola musiman.",
            "Proyeksikan perkembangan ke depan.",
            "Tentukan rekomendasi tindakan preventif.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=AnalisaTrenKejahatan,
        structured_outputs=True,
        debug_mode=False,
    )

    agen_intel: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        instructions=[
            "Lakukan analisis intelijen mendalam:",
            "1. Identifikasi pola operasional pelaku",
            "2. Analisis jejaring dan koneksi antar kasus",
            "3. Evaluasi tingkat ancaman dan risiko",
            "4. Identifikasi indikator awal kejahatan",
            "5. Pemetaan area dan waktu rawan",
        ],
        tools=[GoogleSearchTools(), Newspaper4kTools()],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=False,
    )

    agen_laporan: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        instructions=[
            "Buat laporan analisis kejahatan yang objektif:",
            "1. Ringkasan eksekutif dengan kategori dan deskripsi",
            "2. Analisis detail modus operandi dan bukti dalam format list",
            "3. Analisis temporal dan geografis kejadian",
            "4. Tren dan perkembangan yang teridentifikasi",
            "5. Rekomendasi preemtif dan preventif",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        structured_outputs=True,
        markdown=True,
        debug_mode=False,
    )

    def get_analisis_modus(self, kategori_kejahatan: str) -> Optional[AnalisaPolisional]:
        try:
            response: RunResponse = self.agen_analisis_modus.run(
                f"Lakukan analisis mendalam untuk kategori kejahatan: {kategori_kejahatan}"
            )

            # Check if we got a valid response
            if not response or not response.content:
                logger.warning("Response kosong")
                return None

            # Check if the response is of the expected type
            if not isinstance(response.content, AnalisaPolisional):
                logger.warning("Invalid response type")
                return None

            return response.content

        except Exception as e:
            logger.warning(f"Failed: {str(e)}")
            return None

    def get_analisis_tren(
        self, kategori_kejahatan: str, analisa_modus: AnalisaPolisional
    ) -> Optional[AnalisaTrenKejahatan]:
        agent_input = {
            "kategori_kejahatan": kategori_kejahatan,
            **analisa_modus.model_dump()
        }

        try:
            response: RunResponse = self.agen_analisis_tren.run(
                json.dumps(agent_input, indent=4)
            )

            # Check if we got a valid response
            if not response or not response.content:
                logger.warning("Response kosong")
                return None

            # Check if the response is of the expected type
            if not isinstance(response.content, AnalisaTrenKejahatan):
                logger.warning("Invalid response type")
                return None

            return response.content

        except Exception as e:
            logger.warning(f"Failed: {str(e)}")
            return None

    def get_analisis_intel(
        self, kategori_kejahatan: str, 
        analisa_modus: AnalisaPolisional,
        analisis_tren: AnalisaTrenKejahatan
    ) -> Optional[str]:
        agent_input = {
            "kategori_kejahatan": kategori_kejahatan,
            "data_modus": analisa_modus.model_dump(),
            "data_tren": analisis_tren.model_dump()
        }

        try:
            response: RunResponse = self.agen_intel.run(
                json.dumps(agent_input, indent=4)
            )

            # Check if we got a valid response
            if not response or not response.content:
                logger.warning("Response kosong")
                return None

            return response.content

        except Exception as e:
            logger.warning(f"Failed: {str(e)}")
            return None

    def run(self, kategori_kejahatan: str) -> Iterator[RunResponse]:
        logger.info(f"Memulai analisis intelijen untuk: {kategori_kejahatan}")

        # Step 1: Analisis Modus Operandi
        analisa_modus = self.get_analisis_modus(kategori_kejahatan)
        if analisa_modus is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Gagal menganalisis modus operandi untuk: {kategori_kejahatan}"
            )
            return

        # Step 2: Analisis Tren
        analisis_tren = self.get_analisis_tren(kategori_kejahatan, analisa_modus)
        if analisis_tren is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Gagal menganalisis tren kejahatan"
            )
            return

        # Step 3: Analisis Intelijen
        analisis_intel = self.get_analisis_intel(
            kategori_kejahatan, analisa_modus, analisis_tren
        )

        # Step 4: Compile final report
        final_response: RunResponse = self.agen_laporan.run(
            json.dumps(
                {
                    "kategori_kejahatan": kategori_kejahatan,
                    **analisa_modus.model_dump(),
                    **analisis_tren.model_dump(),
                    "analisis_intelijen": analisis_intel or "Belum ada analisis intelijen"
                },
                indent=4,
            )
        )

        # Return final report and mark workflow as completed
        yield RunResponse(
            content=final_response.content,
            event=RunEvent.workflow_completed
        )

def get_analisator_tren_kejahatan(debug_mode: bool = False) -> SistemAnalisisIntelijen:
    """Create and configure the workflow instance.
    Maintains backward compatibility with existing code."""
    return SistemAnalisisIntelijen(
        workflow_id="analisis-modus-kejahatan",  # Must match the API endpoint path
        description="Sistem Analisis Intelijen Kepolisian",
        storage=redis_storage,
        debug_mode=debug_mode,
    )

# Instantiate workflow if run directly
if __name__ == "__main__":
    from rich.prompt import Prompt
    from agno.utils.pprint import pprint_run_response

    kategori = Prompt.ask(
        "[bold]Masukkan kategori kejahatan untuk dianalisis[/bold]\n✨",
        default="Pencurian kendaraan bermotor"
    )

    url_safe_kategori = kategori.lower().replace(" ", "-")
    
    # Create the workflow with the predefined storage
    analisis_sistem = get_analisator_tren_kejahatan()
    
    # Set a unique session ID for this run
    analisis_sistem.session_id = f"analisis-kejahatan-{url_safe_kategori}"
    
    # Run the workflow
    hasil_analisis = analisis_sistem.run(kategori_kejahatan=kategori)

    # Print the results
    pprint_run_response(hasil_analisis, markdown=True)
