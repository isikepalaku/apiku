import json
from typing import Iterator, Optional, List

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.workflow.postgres import PostgresWorkflowStorage
from agno.tools.googlesearch import GoogleSearch
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
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        instructions=[
            "Identifikasi dan analisis modus operandi kejahatan berdasarkan kategori yang diberikan.",
            "Berikan contoh-contoh kasus nyata untuk setiap modus yang teridentifikasi.",
            "Jelaskan metode dan pola yang digunakan pelaku kejahatan.",
            "Identifikasi target atau sasaran yang sering menjadi korban.",
            "Berikan hasil analisis dalam Bahasa Indonesia yang detail dan terstruktur.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=AnalisaPola,
        structured_outputs=True,
        debug_mode=False,
    )

    agen_analisa_tren: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        tools=[GoogleSearch()],
        instructions=[
            "Analisis tren perubahan modus operandi kejahatan.",
            "Identifikasi faktor-faktor yang mempengaruhi perubahan modus.",
            "Analisis pola musiman dalam penggunaan modus tertentu.",
            "Prediksi kemungkinan perkembangan modus ke depan.",
            "Gunakan data dan kasus terkini untuk mendukung analisis.",
            "Berikan hasil analisis dalam Bahasa Indonesia.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=AnalisaTren,
        structured_outputs=True,
        debug_mode=False,
    )

    agen_rekomendasi: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        tools=[GoogleSearch()],
        instructions=[
            "Berdasarkan analisis modus operandi dan tren:",
            "Sarankan strategi pencegahan untuk setiap modus yang teridentifikasi.",
            "Berikan rekomendasi khusus untuk kelompok target/sasaran.",
            "Usulkan taktik pengamanan untuk lokasi dan waktu rawan.",
            "Sarankan pendekatan investigasi untuk setiap jenis modus.",
            "Berikan rekomendasi dalam Bahasa Indonesia yang praktis dan dapat diterapkan.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=False,
    )

    agen_laporan: Agent = Agent(
        model=OpenAIChat(id=workflow_settings.gpt_4_mini),
        instructions=[
            "Susun laporan komprehensif yang mencakup:",
            "- Daftar dan analisis modus operandi dengan contoh kasus",
            "- Tren dan perkembangan modus operandi",
            "- Rekomendasi pencegahan dan penanganan",
            "Sajikan dalam format yang terstruktur dan mudah dipahami.",
            "Gunakan Bahasa Indonesia yang jelas dan profesional.",
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=False,
    )

    def get_analisa_modus(self, kategori_kejahatan: str) -> Optional[AnalisaPola]:
        try:
            response: RunResponse = self.agen_analisa_modus.run(kategori_kejahatan)

            if not response or not response.content:
                logger.warning("Respons Analisa Modus kosong")
            if not isinstance(response.content, AnalisaPola):
                logger.warning("Tipe respons tidak valid")

            return response.content

        except Exception as e:
            logger.warning(f"Gagal: {str(e)}")

        return None

    def get_analisa_tren(
        self, kategori_kejahatan: str, analisa_pola: AnalisaPola
    ) -> Optional[AnalisaTren]:
        agent_input = {"kategori_kejahatan": kategori_kejahatan, **analisa_pola.model_dump()}

        try:
            response: RunResponse = self.agen_analisa_tren.run(json.dumps(agent_input, indent=4))

            if not response or not response.content:
                logger.warning("Respons Analisa Tren kosong")
            if not isinstance(response.content, AnalisaTren):
                logger.warning("Tipe respons tidak valid")

            return response.content

        except Exception as e:
            logger.warning(f"Gagal: {str(e)}")

        return None

    def get_rekomendasi(self, kategori_kejahatan: str, analisa_tren: AnalisaTren) -> Optional[str]:
        agent_input = {"kategori_kejahatan": kategori_kejahatan, **analisa_tren.model_dump()}

        try:
            response: RunResponse = self.agen_rekomendasi.run(json.dumps(agent_input, indent=4))

            if not response or not response.content:
                logger.warning("Respons Rekomendasi kosong")

            return response.content

        except Exception as e:
            logger.warning(f"Gagal: {str(e)}")

        return None

    def run(self, kategori_kejahatan: str) -> Iterator[RunResponse]:
        logger.info(f"Menganalisis modus operandi untuk kategori: {kategori_kejahatan}")

        analisa_pola: Optional[AnalisaPola] = self.get_analisa_modus(kategori_kejahatan)

        if analisa_pola is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Gagal menganalisis modus operandi untuk kategori: {kategori_kejahatan}",
            )
            return

        analisa_tren: Optional[AnalisaTren] = self.get_analisa_tren(kategori_kejahatan, analisa_pola)

        if analisa_tren is None:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Analisis tren gagal",
            )
            return

        rekomendasi: Optional[str] = self.get_rekomendasi(kategori_kejahatan, analisa_tren)

        final_response: RunResponse = self.agen_laporan.run(
            json.dumps(
                {
                    "kategori_kejahatan": kategori_kejahatan,
                    **analisa_pola.model_dump(),
                    **analisa_tren.model_dump(),
                    "rekomendasi": rekomendasi,
                },
                indent=4,
            )
        )

        yield RunResponse(content=final_response.content, event=RunEvent.workflow_completed)

def get_analisator_tren_kejahatan(debug_mode: bool = False) -> AnalisatorTrenKejahatan:
    return AnalisatorTrenKejahatan(
        workflow_id="analisis-modus-kejahatan",
        storage=PostgresWorkflowStorage(
            table_name="analisa_modus_kejahatan_workflows",
            db_url=db_url,
        ),
        debug_mode=debug_mode,
    )
