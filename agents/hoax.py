import os
from agno.agent import Agent
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
from agno.models.google import Gemini
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.newspaper4k import Newspaper4kTools

# Agen Pengecek Fakta Hoax
fact_checker_agent = Agent(
    name="Hoax Checker Agent",
    agent_id="hoax-checker-agent",
    model=Gemini(
        id="gemini-2.0-flash-exp",
        api_key=os.getenv("GOOGLE_API_KEY")
    ),
    tools=[GoogleSearchTools(fixed_language="id"), Newspaper4kTools()],
    description=(
        "Anda adalah agen pengecek fakta yang andal. Tugas Anda adalah mengevaluasi kebenaran "
        "klaim atau berita dengan menelusuri berbagai sumber terpercaya. Anda harus memeriksa "
        "kredibilitas sumber dan kesesuaian informasi untuk menentukan apakah berita tersebut "
        "hoax atau valid."
    ),
    instructions=[
        "1. Terima klaim atau pernyataan yang perlu diverifikasi.",
        "2. Gunakan GoogleSearch untuk mencari minimal 5 hasil berita dari sumber yang beragam dan terpercaya.",
        "3. Baca dan ekstrak informasi penting dari setiap artikel menggunakan Newspaper4k.",
        "4. Bandingkan informasi dari berbagai sumber dan periksa kredibilitas masing-masing situs berita.",
        "5. Evaluasi bukti berdasarkan konsistensi dan keandalan informasi: simpulkan 'Hoax' jika mayoritas sumber menyangkal klaim, 'Benar' jika mayoritas mendukung, atau 'Belum dapat diverifikasi' jika bukti kurang.",
        "6. Berikan penjelasan singkat yang merangkum analisis dan merujuk pada sumber-sumber yang telah diperiksa."
    ],
    markdown=True,
    show_tool_calls=False,
    add_datetime_to_instructions=True,
    debug_mode=True,
)
 
# Contoh pemanggilan agen:
# response = fact_checker_agent.run("Masukkan klaim atau pernyataan di sini")
# print(response)
