import os
from agno.agent import Agent
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from agno.tools.newspaper4k import Newspaper4kTools

# Agen Fakta Checker
fact_checker_agent = Agent(
    name="Hoax Checker Agent",
    agent_id="hoax-checker-agent",
    model=Gemini(
        id="gemini-2.0-flash-exp",
        api_key=os.getenv("GOOGLE_API_KEY")
    ),
    tools=[TavilyTools(), Newspaper4kTools()],        # Tools yang digunakan
    description=(
        "Anda adalah agen pengecek fakta. Tugas Anda adalah memverifikasi "
        "apakah klaim/berita yang diberikan tergolong hoaks atau belum ada "
        "bukti yang cukup (belum dapat diverifikasi). Anda akan menggunakan "
        "pencarian web dan analisis artikel untuk menyimpulkan keabsahan klaim."
    ),
    instructions=[
        "1. Terima sebuah klaim atau topik yang ingin diverifikasi.",
        "2. Lakukan pencarian untuk menemukan minimal 5 tautan relevan.",
        "3. Baca tiap tautan menggunakan Newspaper4k dan rangkum informasi penting "
        "yang mendukung atau menyanggah klaim.",
        "4. Evaluasi bukti yang terkumpul. Jika Anda menemukan bukti jelas bahwa "
        "berita tersebut salah, simpulkan sebagai 'Hoax'. Jika ada bukti kuat "
        "mendukung kebenarannya, simpulkan 'Benar'. Jika bukti belum memadai "
        "untuk mengambil kesimpulan, simpulkan 'Belum dapat diverifikasi'.",
        "5. Berikan penjelasan singkat mengapa Anda mengambil kesimpulan tersebut."
    ],
    markdown=True,
    show_tool_calls=False,
    add_datetime_to_instructions=True,
    debug_mode=True,
)

# Contoh pemanggilan agen:
# response = fact_checker_agent.run("Masukkan klaim atau pernyataan di sini")
# print(response)
