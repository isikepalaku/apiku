import json
from typing import Iterator, Optional, List, Dict
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.storage.postgres import PostgresStorage
from agno.tools.googlesearch import GoogleSearchTools
from custom_tools.googlescholar import GoogleScholarTools
from agno.tools.jina import JinaReaderTools
from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field

from db.session import db_url
analishukum_agent_storage = PostgresStorage(table_name="ahlihukum_sessions", db_url=db_url)
# Model data dasar
class SumberHukum(BaseModel):
    judul: str = Field(..., description="Judul sumber hukum atau dokumen")
    url: str = Field(..., description="URL sumber dokumen")
    jenis: str = Field(..., description="Jenis sumber (peraturan/putusan/jurnal/artikel)")
    ringkasan: str = Field(..., description="Ringkasan singkat isi dokumen")
    relevansi: str = Field(..., description="Tingkat relevansi dengan topik penelitian")

class PutusanHukum(BaseModel):
    nomor_putusan: str = Field(..., description="Nomor putusan pengadilan")
    jenis_pengadilan: str = Field(..., description="Jenis pengadilan (PN/PT/MA/MK)")
    tahun: str = Field(..., description="Tahun putusan")
    ringkasan_kasus: str = Field(..., description="Ringkasan kasus yang diputus")
    pertimbangan_hukum: str = Field(..., description="Pertimbangan hukum penting")
    amar_putusan: str = Field(..., description="Amar putusan")
    url_sumber: str = Field(..., description="URL sumber putusan")

class StudiKasus(BaseModel):
    judul_kasus: str = Field(..., description="Judul atau nama kasus")
    tahun: str = Field(..., description="Tahun kejadian kasus")
    fakta_penting: str = Field(..., description="Fakta-fakta penting dalam kasus")
    aspek_hukum: str = Field(..., description="Aspek hukum yang dibahas")
    dampak_hukum: str = Field(..., description="Dampak hukum dari kasus ini")
    url_sumber: str = Field(..., description="URL sumber studi kasus")

# Model hasil penelusuran hukum
class HasilPenelitianHukum(BaseModel):
    sumber_literatur: List[SumberHukum] = Field(..., description="Sumber literatur hukum yang ditemukan")
    putusan_relevan: List[PutusanHukum] = Field(..., description="Putusan pengadilan yang relevan")
    studi_kasus: List[StudiKasus] = Field(..., description="Studi kasus yang ditemukan")
    doktrin_hukum: List[str] = Field(..., description="Doktrin-doktrin hukum penting")
    aturan_relevan: List[str] = Field(..., description="Aturan-aturan hukum yang relevan")
    waktu_pengumpulan: str = Field(..., description="Waktu pencarian dilakukan")
    kata_kunci_digunakan: List[str] = Field(..., description="Kata kunci yang digunakan dalam pencarian")

# Model analisis hukum
class AnalisisHukumKomprehensif(BaseModel):
    aspek_normatif: str = Field(..., description="Analisis normatif peraturan terkait")
    aspek_yurisprudensi: str = Field(..., description="Analisis putusan pengadilan terkait")
    aspek_komparatif: str = Field(..., description="Analisis perbandingan dengan sistem lain")
    interpretasi_hukum: str = Field(..., description="Interpretasi dan konstruksi hukum")
    kesenjangan_hukum: str = Field(..., description="Identifikasi kesenjangan regulasi")
    kesimpulan_utama: str = Field(..., description="Kesimpulan utama dari analisis")
    argumentasi_hukum: str = Field(..., description="Basis argumentasi hukum")
    implikasi_praktis: str = Field(..., description="Implikasi praktis dari temuan")
    rekomendasi_kebijakan: str = Field(..., description="Rekomendasi perbaikan kebijakan")
    arah_penelitian: str = Field(..., description="Arah penelitian lanjutan")

# Workflow utama
class SistemPenelitianHukum(Workflow):
    """
    🧠 Sistem Penelitian dan Analisis Hukum Komprehensif

    Workflow ini memungkinkan penelitian hukum mendalam berbasis AI dengan:
    1. Penelusuran sumber hukum dan ekstraksi informasi komprehensif
    2. Analisis hukum normatif, yurisprudensi, dan komparatif
    3. Pelaporan hasil penelitian dalam format formal dan terstruktur

    Kapabilitas utama:
    - Pencarian literatur hukum, putusan pengadilan, dan studi kasus
    - Ekstraksi doktrin, preseden, dan aturan hukum relevan
    - Analisis kesenjangan regulasi dan interpretasi hukum
    - Pembentukan argumentasi hukum koheren
    - Perumusan rekomendasi kebijakan berbasis bukti
    """
    
    penelusuran_hukum_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Lakukan minimal 5 penelusuran hukum untuk topik yang diberikan:",
            "Temukan literatur hukum (textbook, artikel jurnal, peraturan) yang relevan"
        ],
        tools=[GoogleSearchTools()],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=HasilPenelitianHukum,
        debug_mode=False,
    )
    
    analisis_hukum_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Lakukan analisis hukum komprehensif berdasarkan hasil penelusuran:",
            "1. Analisis normatif terhadap peraturan dan aturan yang ditemukan",
            "2. Analisis yurisprudensi dari putusan pengadilan yang relevan",
            "3. Analisis komparatif dengan sistem hukum atau regulasi lain",
            "4. Berikan interpretasi dan konstruksi hukum yang tepat",
            "5. Identifikasi kesenjangan dalam regulasi atau aturan",
            "6. Rumuskan kesimpulan dan argumentasi hukum yang kuat",
            "7. Tentukan implikasi praktis dari temuan penelitian",
            "8. Susun rekomendasi kebijakan yang konkret",
            "9. Sarankan arah penelitian lanjutan yang diperlukan"
        ],
        tools=[JinaReaderTools()],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        response_model=AnalisisHukumKomprehensif,
        debug_mode=False,
    )

    agen_laporan: Agent = Agent(
        model=Gemini(id="gemini-2.0-flash"),
        instructions=[
            "Susun laporan penelitian hukum formal dengan struktur berikut:",
            "1. **Pendahuluan**",
            "   - Latar belakang dan konteks isu hukum",
            "   - Rumusan masalah dan tujuan penelitian",
            "   - Metodologi penelitian hukum yang digunakan",
            "2. **Tinjauan Sumber Hukum**",
            "   - Analisis sumber primer (peraturan, putusan)",
            "   - Evaluasi sumber sekunder (jurnal, doktrin)",
            "   - Konsep-konsep kunci dalam penelitian",
            "3. **Analisis Hukum Komprehensif**",
            "   - Analisis normatif peraturan terkait",
            "   - Analisis yurisprudensi dan kasus hukum",
            "   - Perbandingan hukum dan regulasi",
            "   - Interpretasi dan konstruksi hukum",
            "4. **Kesimpulan dan Rekomendasi**",
            "   - Kesimpulan analitis berdasarkan temuan",
            "   - Argumentasi hukum yang melandasi",
            "   - Implikasi praktis dan teoritis",
            "   - Rekomendasi kebijakan dan regulasi",
            "   - Arah penelitian lanjutan",
            "5. **Referensi Hukum**",
            "   - Daftar peraturan perundang-undangan",
            "   - Yurisprudensi dan putusan pengadilan",
            "   - Sumber sekunder (jurnal, buku, artikel)"
        ],
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        structured_outputs=True,
        markdown=True,
        debug_mode=False,
    )

    def get_cached_penelitian(self, topik_hukum: str) -> Optional[HasilPenelitianHukum]:
        """Mendapatkan hasil penelusuran hukum dari cache jika tersedia."""
        logger.info("Memeriksa cache hasil penelusuran hukum")
        penelitian = self.session_state.get("penelitian_hukum", {}).get(topik_hukum)
        return (
            HasilPenelitianHukum.model_validate(penelitian)
            if penelitian and isinstance(penelitian, dict)
            else None
        )

    def add_penelitian_to_cache(self, topik_hukum: str, hasil_penelusuran: HasilPenelitianHukum):
        """Menyimpan hasil penelusuran hukum ke dalam cache."""
        logger.info(f"Menyimpan hasil penelusuran hukum untuk topik: {topik_hukum}")
        self.session_state.setdefault("penelitian_hukum", {})
        self.session_state["penelitian_hukum"][topik_hukum] = hasil_penelusuran.model_dump()

    def get_cached_analisis(self, topik_hukum: str) -> Optional[AnalisisHukumKomprehensif]:
        """Mendapatkan hasil analisis hukum dari cache jika tersedia."""
        logger.info("Memeriksa cache hasil analisis hukum")
        analisis = self.session_state.get("analisis_hukum", {}).get(topik_hukum)
        return (
            AnalisisHukumKomprehensif.model_validate(analisis)
            if analisis and isinstance(analisis, dict)
            else None
        )

    def add_analisis_to_cache(self, topik_hukum: str, hasil_analisis: AnalisisHukumKomprehensif):
        """Menyimpan hasil analisis hukum ke dalam cache."""
        logger.info(f"Menyimpan hasil analisis hukum untuk topik: {topik_hukum}")
        self.session_state.setdefault("analisis_hukum", {})
        self.session_state["analisis_hukum"][topik_hukum] = hasil_analisis.model_dump()

    def get_cached_laporan(self, topik_hukum: str) -> Optional[str]:
        """Mendapatkan laporan akhir dari cache jika tersedia."""
        logger.info("Memeriksa cache laporan penelitian hukum")
        try:
            cached_data = self.session_state.get("laporan_hukum", {}).get(topik_hukum)
            if cached_data and isinstance(cached_data, str):
                logger.info("Laporan ditemukan di cache")
                return cached_data
            logger.info("Laporan tidak ditemukan di cache atau format tidak valid")
            return None
        except Exception as e:
            logger.error(f"Error saat mengambil laporan dari cache: {str(e)}")
            return None

    def add_laporan_to_cache(self, topik_hukum: str, laporan: str):
        """Menyimpan laporan akhir ke dalam cache."""
        logger.info(f"Menyimpan laporan penelitian hukum untuk topik: {topik_hukum}")
        if not laporan:
            logger.warning("Mencoba menyimpan laporan kosong")
            return
            
        try:
            # Konversi ke string jika bukan string
            laporan_str = str(laporan) if not isinstance(laporan, str) else laporan
            # Pastikan string tidak kosong setelah strip
            laporan_str = laporan_str.strip()
            if not laporan_str:
                logger.warning("Laporan setelah strip kosong, tidak menyimpan ke cache")
                return
                
            # Simpan ke cache
            self.session_state.setdefault("laporan_hukum", {})
            self.session_state["laporan_hukum"][topik_hukum] = laporan_str
            logger.info(f"Laporan berhasil disimpan ke cache untuk topik: {topik_hukum}")
        except Exception as e:
            logger.error(f"Gagal menyimpan laporan ke cache: {str(e)}")

    def run(
        self, 
        topik_hukum: str = None, 
        input: dict = None,
        gunakan_cache_penelusuran: bool = True,
        gunakan_cache_analisis: bool = True,
        gunakan_cache_laporan: bool = True
    ) -> Iterator[RunResponse]:
        """
        Menjalankan workflow penelitian hukum komprehensif.
        
        Args:
            topik_hukum: Topik hukum yang akan diteliti
            input: Dictionary input sebagai alternatif (Swagger API format)
            gunakan_cache_penelusuran: Gunakan hasil penelusuran dari cache jika tersedia
            gunakan_cache_analisis: Gunakan hasil analisis dari cache jika tersedia
            gunakan_cache_laporan: Gunakan laporan final dari cache jika tersedia
        """
        # Mendukung kedua format API: parameter langsung atau melalui 'input'
        if topik_hukum is None and input and isinstance(input, dict):
            topik_hukum = input.get("topik_hukum", "")
            # Cek parameter cache dari input
            if "gunakan_cache_penelusuran" in input:
                gunakan_cache_penelusuran = input.get("gunakan_cache_penelusuran", True)
            if "gunakan_cache_analisis" in input:
                gunakan_cache_analisis = input.get("gunakan_cache_analisis", True)
            if "gunakan_cache_laporan" in input:
                gunakan_cache_laporan = input.get("gunakan_cache_laporan", True)
        
        # Validasi input
        if not topik_hukum:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content="Error: Parameter 'topik_hukum' diperlukan"
            )
            return
        
        logger.info(f"Memulai penelitian hukum untuk: {topik_hukum}")
        
        yield RunResponse(
            event=RunEvent.workflow_started,
            content=f"Memulai penelitian hukum untuk topik: {topik_hukum}"
        )
        
        # Cek apakah laporan sudah ada di cache
        if gunakan_cache_laporan:
            cached_laporan = self.get_cached_laporan(topik_hukum)
            if cached_laporan:
                logger.info(f"Menggunakan laporan dari cache untuk topik: {topik_hukum}")
                # Pastikan format respon cache valid
                try:
                    yield RunResponse(
                        content=cached_laporan,
                        event=RunEvent.workflow_completed
                    )
                    return
                except Exception as e:
                    logger.error(f"Error saat mengembalikan cache: {str(e)}")
                    # Jika terjadi kesalahan, lanjutkan tanpa cache
        
        # Step 1: Penelusuran sumber hukum dan ekstraksi informasi
        hasil_penelusuran = None
        if gunakan_cache_penelusuran:
            hasil_penelusuran = self.get_cached_penelitian(topik_hukum)
            if hasil_penelusuran:
                logger.info(f"Menggunakan hasil penelusuran dari cache untuk topik: {topik_hukum}")
        
        if hasil_penelusuran is None:
            yield RunResponse(
                event=RunEvent.run_started,
                content=f"Melakukan penelusuran sumber hukum untuk topik: {topik_hukum}"
            )
            
            try:
                # Penelusuran baru tanpa cache
                response: RunResponse = self.penelusuran_hukum_agent.run(
                    f"Lakukan penelusuran hukum komprehensif untuk topik: {topik_hukum}"
                )
                
                if not response or not response.content or not isinstance(response.content, HasilPenelitianHukum):
                    yield RunResponse(
                        event=RunEvent.workflow_completed,
                        content=f"Gagal melakukan penelusuran hukum untuk: {topik_hukum}"
                    )
                    return
                
                hasil_penelusuran = response.content
                self.add_penelitian_to_cache(topik_hukum, hasil_penelusuran)
                
                yield RunResponse(
                    event=RunEvent.run_completed,
                    content=f"Penelusuran selesai. Menemukan {len(hasil_penelusuran.sumber_literatur)} sumber, {len(hasil_penelusuran.putusan_relevan)} putusan, dan {len(hasil_penelusuran.studi_kasus)} studi kasus."
                )
            except Exception as e:
                logger.error(f"Error dalam penelusuran hukum: {str(e)}")
                yield RunResponse(
                    event=RunEvent.workflow_completed,
                    content=f"Gagal melakukan penelusuran hukum: {str(e)}"
                )
                return
        
        # Step 2: Analisis hukum komprehensif
        analisis_hukum = None
        if gunakan_cache_analisis:
            analisis_hukum = self.get_cached_analisis(topik_hukum)
            if analisis_hukum:
                logger.info(f"Menggunakan hasil analisis dari cache untuk topik: {topik_hukum}")
        
        if analisis_hukum is None:
            yield RunResponse(
                event=RunEvent.run_started,
                content="Melakukan analisis hukum komprehensif dari sumber-sumber yang ditemukan..."
            )
            
            try:
                # Analisis baru tanpa cache
                agent_input = {
                    "topik_hukum": topik_hukum,
                    **hasil_penelusuran.model_dump()
                }
                
                response: RunResponse = self.analisis_hukum_agent.run(
                    json.dumps(agent_input, indent=4)
                )
                
                if not response or not response.content or not isinstance(response.content, AnalisisHukumKomprehensif):
                    yield RunResponse(
                        event=RunEvent.workflow_completed,
                        content="Gagal melakukan analisis hukum komprehensif"
                    )
                    return
                
                analisis_hukum = response.content
                self.add_analisis_to_cache(topik_hukum, analisis_hukum)
                
                yield RunResponse(
                    event=RunEvent.run_completed,
                    content="Analisis hukum komprehensif selesai. Menyusun laporan final..."
                )
            except Exception as e:
                logger.error(f"Error dalam analisis hukum: {str(e)}")
                yield RunResponse(
                    event=RunEvent.workflow_completed,
                    content=f"Gagal melakukan analisis hukum: {str(e)}"
                )
                return
        
        # Step 3: Penyusunan laporan final
        yield RunResponse(
            event=RunEvent.run_started,
            content="Menyusun laporan penelitian hukum final..."
        )
        
        try:
            agent_input = {
                "topik_hukum": topik_hukum,
                "hasil_penelusuran": hasil_penelusuran.model_dump(),
                "analisis_hukum": analisis_hukum.model_dump()
            }
            
            # Jalankan agen laporan akhir
            laporan_response = self.agen_laporan.run(json.dumps(agent_input, indent=4))
            
            if not laporan_response or not laporan_response.content:
                yield RunResponse(
                    event=RunEvent.workflow_completed,
                    content="Gagal menyusun laporan final"
                )
                return
            
            # Pastikan konten adalah string dan simpan ke cache
            try:
                laporan_content = laporan_response.content
                # Convert to string if not already a string
                if not isinstance(laporan_content, str):
                    logger.warning(f"Laporan bukan string, mencoba konversi: {type(laporan_content)}")
                    laporan_content = str(laporan_content)
                
                # Validasi dan simpan ke cache
                if laporan_content.strip():
                    self.add_laporan_to_cache(topik_hukum, laporan_content)
                else:
                    logger.warning("Laporan kosong setelah strip, tidak menyimpan ke cache")
            except Exception as e:
                logger.error(f"Error saat memproses laporan untuk cache: {str(e)}")
            
            # Kembalikan hasil akhir dengan format yang konsisten
            yield RunResponse(
                event=RunEvent.run_completed,
                content="Laporan penelitian hukum telah selesai disusun"
            )
            
            # Berikan output akhir dengan penanganan error yang baik
            try:
                # Pastikan output adalah string valid
                final_content = laporan_response.content
                if not isinstance(final_content, str):
                    final_content = str(final_content)
                
                # Pastikan tidak kosong
                if not final_content.strip():
                    final_content = "Laporan kosong, terjadi kesalahan dalam pemrosesan konten"
                
                # Yield final response
                yield RunResponse(
                    content=final_content,
                    event=RunEvent.workflow_completed
                )
            except Exception as e:
                logger.error(f"Error saat memformat respon akhir: {str(e)}")
                yield RunResponse(
                    content=f"Error: Format output tidak valid. Detail: {str(e)}",
                    event=RunEvent.workflow_completed
                )
            
        except Exception as e:
            logger.error(f"Error dalam penyusunan laporan final: {str(e)}")
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"Gagal menyusun laporan final: {str(e)}"
            )

def get_sistem_penelitian_hukum(debug_mode: bool = False, user_id: str = None, session_id: str = None) -> SistemPenelitianHukum:
    """Membuat dan mengkonfigurasi instance workflow penelitian hukum."""
    
    # Buat session_id yang aman jika disediakan
    safe_session_id = f"penelitian-hukum-{session_id}" if session_id else "penelitian-hukum"
    
    workflow = SistemPenelitianHukum(
        workflow_id="sistem-penelitian-hukum",  # Must match the API endpoint path
        description="Sistem Penelitian dan Analisis Hukum Komprehensif",
        session_id=safe_session_id,
        user_id=user_id,
        storage=analishukum_agent_storage,
    )

    if debug_mode:
        logger.info("Mode debug aktif untuk semua agen")
        workflow.penelusuran_hukum_agent.debug_mode = True
        workflow.analisis_hukum_agent.debug_mode = True
        workflow.agen_laporan.debug_mode = True

    return workflow

# Jalankan workflow jika dieksekusi sebagai script
if __name__ == "__main__":
    from rich.prompt import Prompt
    import uuid

    topik = Prompt.ask(
        "[bold]Masukkan topik hukum untuk diteliti[/bold]\n✨",
        default="Perlindungan data pribadi dalam ekonomi digital"
    )

    url_safe_topik = topik.lower().replace(" ", "-")
    
    # Gunakan uuid untuk session_id unik
    session_id = f"{url_safe_topik}-{str(uuid.uuid4())[:8]}"

    sistem_penelitian = get_sistem_penelitian_hukum(
        debug_mode=True,
        session_id=session_id,
        user_id="local-user"
    )

    # Jalankan workflow dengan parameter caching tambahan
    hasil_penelitian = sistem_penelitian.run(
        topik_hukum=topik,
        gunakan_cache_penelusuran=True,
        gunakan_cache_analisis=True,
        gunakan_cache_laporan=False  # Selalu buat laporan baru
    )

    from agno.utils.pprint import pprint_run_response
    pprint_run_response(hasil_penelitian, markdown=True) 